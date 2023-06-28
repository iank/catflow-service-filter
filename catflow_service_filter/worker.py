from typing import Any, List, Tuple
import signal
import asyncio
from catflow_worker import Worker

import logging


async def filter_handler(
    msg: Any, key: str, s3: Any, bucket: str
) -> Tuple[bool, List[Tuple[str, Tuple[str, List[float]]]]]:
    """Run filter on the given frames"""
    logging.info(f"[*] Message received ({key})")

    return True, [("filter.rawframes", [])]


async def shutdown(worker, task):
    await worker.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.exceptions.CancelledError:
        pass


async def startup(queue: str, topic_key: str):
    worker = await Worker.create(filter_handler, queue, topic_key)
    task = asyncio.create_task(worker.work())

    def handle_sigint(sig, frame):
        print("^ SIGINT received, shutting down...")
        asyncio.create_task(shutdown(worker, task))

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if not await task:
            print("[!] Exited with error")
            return False
    except asyncio.exceptions.CancelledError:
        return True


def main() -> bool:
    topic_key = "filter.embeddings"
    queue_name = "catflow-service-filter"
    logging.basicConfig(level=logging.INFO)

    return asyncio.run(startup(queue_name, topic_key))
