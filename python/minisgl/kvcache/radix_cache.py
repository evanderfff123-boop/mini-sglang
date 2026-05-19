from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]  # 用于从 token id 序列生成节点键的函数类型


class RadixTreeNode:
    """基数树节点，代表一段连续的 token 序列及其缓存位置"""
    counter: int = 0  # 全局节点计数器，用于生成唯一 UUID

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn  # 键值生成函数
        self.children: Dict[Any, RadixTreeNode] = {}  # 子节点字典，键为第一个 page 的 token 表示
        self._parent: RadixTreeNode | None = None  # 父节点
        self.ref_count: int = 0  # 引用计数，>0 表示被锁定不可驱逐
        self.uuid = RadixTreeNode.counter  # 唯一标识符
        RadixTreeNode.counter += 1  # 递增全局计数器
        self.timestamp = tic or time.monotonic_ns()  # 最近访问时间戳，用于 LRU 驱逐

        # 这些字段应在之后通过 set_key_value 更新
        self._key: torch.Tensor  # 节点覆盖的 token id 序列
        self._value: torch.Tensor  # 节点覆盖的缓存索引（page table 索引）
        self._length: int  # 节点长度

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置节点的 key（token id）和 value（缓存索引）"""
        assert len(key) == len(value)
        self._key = key  # token id 序列
        self._value = value  # 对应的缓存索引序列
        self._length = len(key)  # 序列长度

    def set_parent(self, parent: RadixTreeNode) -> None:
        """设置父节点，并将自身注册到父节点的 children 字典中"""
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self  # 以 key_fn 的结果作为子节点键

    @property
    def length(self) -> int:
        return self._length  # 节点覆盖的 token 数量

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent  # 父节点

    @property
    def value(self) -> torch.Tensor:
        return self._value  # 缓存索引序列

    def is_root(self) -> bool:
        return self._parent is None  # 是否为根节点

    def is_leaf(self) -> bool:
        return len(self.children) == 0  # 是否为叶节点

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """计算当前节点与输入序列的公共前缀长度"""
        from minisgl.kernel import fast_compare_key

        # 比较节点 key 和输入序列，找到第一个不同位置
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在 pos 位置将当前节点一分为二，返回新的前缀节点"""
        assert 0 < pos < self.length
        parent = self.parent  # 原父节点

        # 创建新节点，持有前半段
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])  # 前半段 key 和 value
        new_node.set_parent(parent)  # 新节点挂在原父节点下
        new_node.ref_count = self.ref_count  # 继承原引用计数

        # 当前节点变为后半段
        self.set_key_value(self._key[pos:], self._value[pos:])  # 后半段 key 和 value
        self.set_parent(new_node)  # 当前节点挂在新的前缀节点下

        return new_node  # 返回新的前缀节点

    def __lt__(self, other: RadixTreeNode) -> bool:
        """比较操作符，用于堆排序（按时间戳）"""
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """基数树缓存句柄，记录匹配到的节点引用"""
    node: RadixTreeNode  # 匹配到的节点

    def get_matched_indices(self) -> torch.Tensor:
        """从匹配节点向上遍历到根节点，收集所有缓存索引"""
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():  # 从当前节点遍历到根
            value_list.append(node.value)  # 收集每个节点的 value（缓存索引）
            node = node.parent
        value_list.reverse()  # 反转得到从根到叶的顺序
        return torch.cat(value_list)  # 拼接为完整索引序列


class RadixPrefixCache(BasePrefixCache):
    """基于基数树的前缀缓存，实现前缀共享和 LRU 驱逐"""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device  # 缓存设备
        self.page_size = get_global_ctx().page_size  # 页面大小
        self.key_fn = _get_key_fn(self.page_size)  # 获取适用于当前 page_size 的键函数
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)  # 空张量，用于 evict(0)
        self.evictable_size = 0  # 可驱逐的缓存大小
        self.protected_size = 0  # 被保护的缓存大小
        self.root_node = RadixTreeNode(self.key_fn)  # 根节点
        self.root_node.ref_count = 1  # 根节点始终被保护，不可驱逐

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁定或解锁缓存句柄，更新引用计数和大小统计"""
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            # 解锁：从当前节点向上递减 ref_count
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:  # 引用归零后变为可驱逐
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            # 锁定：从当前节点向上递增 ref_count
            while not node.is_root():
                if node.ref_count == 0:  # 从可驱逐变为被保护
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """在基数树中匹配输入序列的前缀"""
        node, prefix_len = self._tree_walk(input_ids)  # 在树中行走查找最长匹配
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """将输入序列的缓存索引插入基数树"""
        insert_len = align_down(len(input_ids), self.page_size)  # 按 page_size 对齐长度
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]  # 截断到对齐长度
        node, prefix_len = self._tree_walk(input_ids)  # 先匹配已存在的前缀
        if prefix_len != insert_len:  # 说明 prefix_len < insert_len，需要创建新节点
            new_node = RadixTreeNode(self.key_fn)  # 创建新节点
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())  # 设置剩余部分
            new_node.set_parent(node)  # 挂在匹配到的节点下
            self.evictable_size += new_node.length  # 新节点默认可驱逐
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """驱逐指定大小的缓存，使用 LRU 策略"""
        if size == 0:
            return self.empty_tensor  # 请求大小为0，直接返回空
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()  # 收集所有可驱逐的叶节点
        heapq.heapify(leave_nodes)  # 按时间戳建最小堆（最早访问的先被驱逐）
        evicted_indices: List[torch.Tensor] = []  # 被驱逐的索引列表
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)  # 取出最久未访问的叶节点
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length  # 累加被驱逐的大小
            evicted_indices.append(node.value)  # 记录被驱逐的缓存索引
            self.evictable_size -= node.length  # 更新可驱逐大小
            parent = node.parent
            del parent.children[self.key_fn(node._key)]  # 从父节点的 children 中移除
            # 注意：根节点始终被保护，不会被驱逐
            if parent.is_leaf() and parent.ref_count == 0:  # 父节点变为叶节点且可驱逐
                heapq.heappush(leave_nodes, parent)  # 加入可驱逐堆

        return torch.cat(evicted_indices)  # 拼接所有被驱逐的索引

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,  # 可驱逐缓存大小
            protected_size=self.protected_size,    # 被保护缓存大小
        )

    def check_integrity(self) -> None:
        pass  # 暂时不实现完整性检查

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """收集所有引用计数为0的叶节点（可被驱逐的候选节点）"""
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():  # 叶节点
                if node.ref_count == 0:  # 未被引用（可驱逐）
                    leave_nodes.append(node)
            else:
                for child in node.children.values():  # 遍历子节点
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """在基数树中逐层匹配输入序列，返回最终节点和已匹配长度"""
        prefix_len = 0  # 已匹配的前缀长度
        indice_len = len(input_ids)  # 输入序列总长度
        node = self.root_node  # 从根节点开始
        tic = time.monotonic_ns()  # 当前时间戳，用于更新访问时间

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))  # 查找匹配的子节点
            if child_node is None:  # 没有匹配的子节点
                return node, prefix_len
            node = child_node  # 移动到子节点继续匹配

            # 注意：至少匹配 1 个 page，所以 match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])  # 计算与当前节点的匹配长度
            match_len = align_down(match_len, self.page_size)  # 按 page_size 向下取整
            prefix_len += match_len  # 更新已匹配长度

            # 如果未完全匹配当前节点，则需要拆分
            if match_len != node.length:
                node = node.split_at(match_len)  # 在匹配位置拆分节点
                node.timestamp = tic  # 更新时间戳
                return node, prefix_len

            # 完全匹配当前节点，更新时间戳（LRU 策略）
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    """根据 page_size 生成对应的键函数"""
    if page_size == 1:
        return lambda x: x[0].item()  # page_size=1 时，用第一个 token 的值作为键
    return lambda x: tuple(x[:page_size].tolist())  # page_size>1 时，用前 page_size 个 token 的元组作为键
