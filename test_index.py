import pytest

from src.index import root, say_hello, hello_message


@pytest.mark.asyncio
async def test_root():
    result = await root()
    assert result == {'message': 'Hello World'}


@pytest.mark.asyncio
async def test_say_hello():
    result = await say_hello("John")
    assert result == {'message': 'Hello John'}
