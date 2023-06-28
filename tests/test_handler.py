import pytest
from catflow_service_filter.worker import filter_handler


@pytest.mark.asyncio
async def test_worker():
    # Test worker's behavior
    filter_handler()  # TODO
    pass
