import torch


class TableManager:
    """管理请求的行表（token 池和页表）的分配与释放。"""

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs  # 最大并发请求数
        self._free_slots = list(range(max_running_reqs))  # 空闲的行表槽位列表
        self.page_table = page_table  # 页表（[num_reqs, max_seq_len]）
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)  # token 池，用于存储每个位置的实际 token ID

    @property
    def available_size(self) -> int:
        """当前空闲的行表槽位数。"""
        return len(self._free_slots)

    def allocate(self) -> int:
        """分配一个空闲槽位，返回其索引。"""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """释放指定的槽位，归还到空闲列表。"""
        self._free_slots.append(slot)
