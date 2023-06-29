from typing import Any, List, Tuple
import signal
import asyncio
from catflow_worker import Worker
from catflow_worker.types import (
    Embedding,
    RawFrameSchema,
    EmbeddedFrameSchema,
    RawFrame,
)
import numpy as np
import psycopg2
import os
from pgvector.psycopg2 import register_vector

import logging


class VDB:
    """DB ops"""

    def __init__(self, pgconfig, threshold):
        self.threshold = threshold

        self.db = psycopg2.connect(
            host=pgconfig["host"],
            dbname=pgconfig["dbname"],
            user=pgconfig["username"],
            password=pgconfig["password"],
            port=pgconfig["port"],
        )
        register_vector(self.db)

    def check_novel(self, embedding: Embedding):
        embedding = np.array(embedding.vector)

        # Check nearest
        query = (
            "SELECT uuid, cosine_distance(embedding, %s) as distance"
            " FROM images ORDER BY distance LIMIT 1"
        )
        cur = self.db.cursor()
        cur.execute(query, (embedding,))
        results = cur.fetchall()
        cur.close()

        if len(results) == 0:
            return True

        (nearest_uuid, min_distance) = results[0]  # first/only result
        logging.debug(
            f"cosine distance to {nearest_uuid} (nearest): {min_distance:.4f}"
        )
        if min_distance < self.threshold:
            return False

        return True

    def add(self, uuid: str, source: str, embedding: Embedding):
        embedding = np.array(embedding.vector)
        cur = self.db.cursor()
        insertQuery = "INSERT INTO images (uuid, embedding, source) VALUES (%s, %s, %s)"
        cur.execute(insertQuery, (uuid, embedding, source))
        self.db.commit()
        cur.close()


def create_filter_handler(db):
    async def _filter_handler(
        msg: Any, key: str, s3: Any, bucket: str
    ) -> Tuple[bool, List[Tuple[str, Any]]]:
        """Run filter on the given frames"""
        logging.info(f"[*] Message received ({key})")

        # Load message
        embedded_frames = EmbeddedFrameSchema(many=True).load(msg)

        # Check frames and add them to DB one by one
        responseobjects = []
        for frame in embedded_frames:
            embedding = frame.embedding

            if db.check_novel(embedding):
                uuid, ext = frame.key.split(".")
                db.add(uuid, frame.source.key, embedding)
                rawframe = RawFrame(key=frame.key, source=frame.source)
                logging.info(f"[-] Adding {rawframe!s} to database")
                responseobjects.append(rawframe)

        # Empty response if no frames added
        if len(responseobjects) == 0:
            return True, []

        # Dump response
        schema_out = RawFrameSchema(many=True)
        responseobjects_msg = schema_out.dump(responseobjects)

        logging.info(f"[-] {len(responseobjects)} objects -> filter.rawframes (1 msg)")
        return True, [("filter.rawframes", responseobjects_msg)]

    return _filter_handler


async def shutdown(worker, task):
    await worker.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.exceptions.CancelledError:
        pass


async def startup(queue: str, topic_key: str):
    # Set up DB
    pgconfig = {
        "host": os.environ["CATFLOW_PG_HOST"],
        "dbname": os.environ["CATFLOW_PG_DBNAME"],
        "username": os.environ["CATFLOW_PG_USER"],
        "password": os.environ["CATFLOW_PG_PASSWORD"],
        "port": int(os.environ["CATFLOW_PG_PORT"]),
    }

    threshold = float(os.environ["CATFLOW_VDB_THRESHOLD"])

    db = VDB(pgconfig, threshold)
    filter_handler = create_filter_handler(db)

    # Start worker
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
