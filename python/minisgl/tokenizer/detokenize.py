from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# 代码参考自 sglang


def _is_chinese_char(cp: int):
    """判断码点 cp 是否属于 CJK 统一表意文字区块"""
    # 定义中文为 CJK 统一表意文字区块中的字符：
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # 注意：CJK 区块并不包含所有日文和韩文字符。
    # 现代韩文属于不同的区块，日文平假名和片假名同理。
    # 这些语言的单词以空格分隔，因此不特殊处理。
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """返回 text 中最长的可打印子串，保证只包含完整的词。"""
    # 代码参考自 https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # 遇到换行符时，直接返回整个字符串（刷新缓存）
    if text.endswith("\n"):
        return text
    # 如果最后一个 token 是中文，直接打印所有字符
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # 如果倒数第二个 token 是中文，则去掉最后一个字符再打印
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # 其他情况：打印到最后一个空格之前
    # （简单启发式方法，避免打印不完整的词，下一个 token 可能补全它）
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    """解码状态：跟踪某个 uid 的解码进度"""

    decoded_ids: List[int]  # 已收到的所有 token ID 列表
    decoded_str: str  # 已解码的累积文本
    read_offset: int  # 已读取的 ID 位置
    surr_offset: int  # 已确认的 ID 位置（前一个稳定输出的末尾）
    sent_offset: int  # 已发送给前端的文本长度


class DetokenizeManager:
    """文本解码管理器：将 token ID 流逐步解码为可读文本"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus 映射
        self.decode_map: Dict[int, DecodeStatus] = {}  # 每个 uid 的解码状态
        self.tokenizer = tokenizer  # HuggingFace tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id  # 结束符 token ID

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """批量解码，返回每个消息的增量文本"""
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                # 首次遇到该 uid，初始化解码状态
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)  # 追加新生成的 token（排除结束符）
            read_ids.append(s.decoded_ids[s.surr_offset :])  # 自上一个确认位置以来的所有 ID
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])  # 已确认的 ID 序列

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]  # 新解码出的文本部分
            # 流式 chunk 处理：更新解码状态
            if len(new_text) > 0 and not new_text.endswith("�"):
                # 如果新文本有效且不以替换字符结尾（即解码稳定），更新确认位置
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str  # 更新累积文本
                s.surr_offset = s.read_offset  # 确认位置前移到当前读取位置
                s.read_offset = len(s.decoded_ids)  # 读取位置前进
            else:
                # 新文本可能不稳定（被截断），用启发式方法找到可打印部分
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            incremental_output = output_str[s.sent_offset :]  # 自上次发送以来的增量
            s.sent_offset = len(output_str)  # 更新已发送位置
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]  # 请求结束，清理解码状态

        return incremental_strs
