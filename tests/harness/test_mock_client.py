import pytest

from atlas_rl.harness.mock_client import MockLLMClient, ScriptedResponse


@pytest.mark.asyncio
async def test_mock_client_returns_scripted_responses_in_order():
    client = MockLLMClient(
        scripted=[
            ScriptedResponse(text='{"action": "NORTH"}', tokens_in=100, tokens_out=5),
            ScriptedResponse(text='{"action": "EAST"}', tokens_in=120, tokens_out=5),
        ]
    )
    r1 = await client.complete("sys", "user1", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)
    r2 = await client.complete("sys", "user2", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)
    assert r1.text == '{"action": "NORTH"}'
    assert r1.tokens_in == 100
    assert r2.text == '{"action": "EAST"}'


@pytest.mark.asyncio
async def test_mock_client_raises_when_script_exhausted():
    client = MockLLMClient(scripted=[ScriptedResponse(text='{"action": "NORTH"}')])
    await client.complete("s", "u", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)
    with pytest.raises(RuntimeError, match="exhausted"):
        await client.complete("s", "u", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)


@pytest.mark.asyncio
async def test_mock_client_cost_is_zero():
    client = MockLLMClient(scripted=[ScriptedResponse(text='{"action": "NORTH"}')])
    r = await client.complete("s", "u", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)
    assert r.dollar_cost == 0.0
    assert r.provider == "mock"


@pytest.mark.asyncio
async def test_mock_client_always_same_response_mode():
    client = MockLLMClient.always('{"action": "NOOP"}')
    for _ in range(10):
        r = await client.complete("s", "u", temperature=0.0, max_output_tokens=100, response_format=None, seed=None)
        assert r.text == '{"action": "NOOP"}'
