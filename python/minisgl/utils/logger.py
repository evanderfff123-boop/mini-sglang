from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None  # 全局缓存日志级别


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """初始化带颜色格式化输出的日志器。"""

    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()  # 优先从环境变量读取日志级别
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)  # 默认 INFO

    if strip_file:
        suffix = os.path.basename(suffix)  # 只保留文件名部分

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")  # 环境变量控制是否显示 PID

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # 带颜色的格式化器类
    class ColorFormatter(logging.Formatter):
        """提供彩色和美化输出的日志格式化器"""

        # ANSI 颜色代码
        COLORS = {
            "DEBUG": "\033[36m",  # 青色
            "INFO": "\033[32m",  # 绿色
            "WARNING": "\033[33m",  # 黄色
            "ERROR": "\033[31m",  # 红色
            "CRITICAL": "\033[35m",  # 洋红色
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            from minisgl.distributed import try_get_tp_info

            # 格式化时间戳，类似 SGLang 风格: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"  # 追加 TP rank 信息
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # 获取日志级别对应的颜色
            level_color = self.COLORS.get(record.levelname, "")

            # 格式化消息
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # 美化格式: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # 清空已有 handler 避免重复
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)  # 输出到 stdout
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 阻止日志传播到根 logger
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        """只在主 rank (rank 0) 上执行的日志方法"""
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """自定义日志器包装器，提供可选的 rank0 方法签名（仅用于类型提示）"""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        # 为 logger 动态添加 rank0 方法
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger
