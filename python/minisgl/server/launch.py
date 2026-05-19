from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """在子进程中运行 scheduler 主循环"""
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()  # 等待所有 TP rank 就绪

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")  # 主 rank 通知启动器 scheduler 已准备就绪

        if args.silent_output:
            logging.disable(logging.INFO)  # 静默模式下关闭 INFO 日志

        try:
            scheduler.run_forever()  # 进入主循环，持续处理请求
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # 在 ^C 后输出一个空行保持格式整洁
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    """启动完整的推理服务：包括 scheduler 进程、tokenizer 进程和前端 API 服务器"""
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        """启动 scheduler 和 tokenizer 子进程"""
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)  # 使用 spawn 方式启动子进程（兼容 CUDA）

        world_size = server_args.tp_info.size
        # 多进程队列，用于接收子进程的就绪确认
        ack_queue: mp.Queue[str] = mp.Queue()

        # 为每个 TP rank 启动一个 scheduler 进程
        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),  # 每个 rank 有独立的 tp_info
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # 启动 detokenizer 进程（只有 1 个）
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # 启动独立的 tokenizer 进程（如果 num_tokenizer > 0）
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # 等待所有工作进程发送就绪确认：
        # - world_size 个 scheduler（但只有主 rank 发送确认）
        # - num_tokenizers 个 tokenizer
        # - 1 个 detokenizer
        # 总共确认数：1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
