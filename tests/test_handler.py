import pytest
from catflow_service_filter.worker import create_filter_handler
from catflow_worker.types import (
    Embedding,
    EmbeddedFrame,
    EmbeddedFrameSchema,
    RawFrameSchema,
    VideoFile,
)
import numpy as np


@pytest.fixture()
def vdb():
    return VDB_Mock(threshold=0.2)


class VDB_Mock:
    """Fake DB ops rather than stand up a pgvector DB for unit tests"""

    def __init__(self, threshold):
        self.threshold = threshold
        self.database = {}

    def cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def check_novel(self, embedding: Embedding):
        # filter
        distances = [
            self.cosine_distance(x, embedding.vector) for x in self.database.values()
        ]
        if len(distances) == 0:
            return True

        closest = np.min(distances)
        if closest < self.threshold:
            return False
        else:
            return True

    def add(self, uuid: str, source: str, embedding: Embedding):
        self.database[uuid] = embedding.vector


@pytest.mark.asyncio
async def test_worker_correct_output(vdb):
    filter_handler = create_filter_handler(vdb)

    # Expected input: list of EmbeddedFrame
    video = VideoFile(key="test.mp4")
    frames = [
        EmbeddedFrame(
            key="test1.png", source=video, embedding=Embedding(vector=[1.0, -1.0, 3])
        ),
    ]
    embeddings_msg = EmbeddedFrameSchema(many=True).dump(frames)

    # Expected output: list of RawFrame
    output_schema = RawFrameSchema(many=True)

    status, responses = await filter_handler(
        embeddings_msg, "filter.embeddings", None, None
    )

    assert status is True
    assert len(responses) == 1

    routing_key, frames_msg = responses[0]
    frames = output_schema.load(frames_msg)

    assert routing_key == "filter.rawframes"
    assert len(frames) == 1

    frame = frames[0]
    assert frame.key == "test1.png"
    assert frame.source.key == "test.mp4"


@pytest.mark.asyncio
async def test_worker_reject_close_embedding(vdb):
    filter_handler = create_filter_handler(vdb)

    # Add an embedding to DB
    vdb.add("test.png", "", Embedding(vector=[1.0, -1.0, 3.1]))

    # Expected input: list of EmbeddedFrame
    video = VideoFile(key="test.mp4")
    frames = [
        EmbeddedFrame(
            key="tooclose.png", source=video, embedding=Embedding(vector=[1.0, -1.0, 3])
        ),
    ]
    embeddings_msg = EmbeddedFrameSchema(many=True).dump(frames)

    status, responses = await filter_handler(
        embeddings_msg, "filter.embeddings", None, None
    )

    # test that close embedding gets rejected
    assert status is True
    assert len(responses) == 0


@pytest.mark.asyncio
async def test_worker_add_far_embedding(vdb):
    filter_handler = create_filter_handler(vdb)

    # Add an embedding to DB
    vdb.add("test.png", "", Embedding(vector=[1.0, -1.0, 3.1]))

    # Expected input: list of EmbeddedFrame
    video = VideoFile(key="test.mp4")
    frames = [
        EmbeddedFrame(
            key="novel.png", source=video, embedding=Embedding(vector=[1.0, 1.0, 1.0])
        ),
        EmbeddedFrame(
            key="similar_to_the_first_one.png",
            source=video,
            embedding=Embedding(vector=[1.0, 1.0, 1.1]),
        ),
    ]
    embeddings_msg = EmbeddedFrameSchema(many=True).dump(frames)

    # Expected output: list of RawFrame
    output_schema = RawFrameSchema(many=True)

    status, responses = await filter_handler(
        embeddings_msg, "filter.embeddings", None, None
    )

    assert status is True
    assert len(responses) == 1

    routing_key, frames_msg = responses[0]
    frames = output_schema.load(frames_msg)

    # Test that the first frame was added but not the second
    assert frames[0].key == "novel.png"

    # verify that it was actually added to DB
    assert "novel" in vdb.database
    assert "similar_to_the_first_one" not in vdb.database


@pytest.mark.asyncio
async def test_worker_add_uuid_only(vdb):
    filter_handler = create_filter_handler(vdb)

    # Expected input: list of EmbeddedFrame
    video = VideoFile(key="test.mp4")
    frames = [
        EmbeddedFrame(
            key="d4345b64-1e80-44b1-9e51-828c51325fd0.png",
            source=video,
            embedding=Embedding(vector=[1.0, 1.0, 1.0]),
        ),
    ]
    embeddings_msg = EmbeddedFrameSchema(many=True).dump(frames)

    status, responses = await filter_handler(
        embeddings_msg, "filter.embeddings", None, None
    )

    assert status is True
    assert len(responses) == 1

    assert "d4345b64-1e80-44b1-9e51-828c51325fd0" in vdb.database
    assert "d4345b64-1e80-44b1-9e51-828c51325fd0.png" not in vdb.database
