from gateway.stats.capture import SseUsageScanner, Usage, extract_usage


def test_openai_nonstream():
    body = {"choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 40}}}
    assert extract_usage(body) == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_openai_cached_tokens_at_top_level():
    body = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                     "cached_tokens": 40}}
    assert extract_usage(body) == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_openai_deepseek_style_cache_hit():
    body = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                     "prompt_cache_hit_tokens": 40}}
    assert extract_usage(body) == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_openai_prompt_details_cached_creation():
    body = {"choices": [], "usage": {"prompt_tokens": 2928, "completion_tokens": 1,
                                     "prompt_tokens_details": {"cached_tokens": 0,
                                                               "cached_creation_tokens": 2926}}}
    assert extract_usage(body) == Usage(input_tokens=2, cache_read_tokens=0,
                                        cache_write_tokens=2926, output_tokens=1)


def test_openai_top_level_cached_creation():
    body = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                     "cached_creation_tokens": 30}}
    assert extract_usage(body) == Usage(input_tokens=70, cache_read_tokens=0,
                                        cache_write_tokens=30, output_tokens=20)


def test_openai_prefers_openai_keys_when_both_present():
    # Relays (e.g. buzzai) emit both key families; OpenAI keys must win when
    # the response is OpenAI-style (prompt_tokens present or usage_semantic).
    body = {"choices": [], "usage": {
        "prompt_tokens": 2928, "completion_tokens": 1, "total_tokens": 2929,
        "usage_semantic": "openai",
        "prompt_tokens_details": {"cached_tokens": 0, "cached_creation_tokens": 2926},
        "input_tokens": 2928, "output_tokens": 0,
        "claude_cache_creation_5_m_tokens": 2926}}
    assert extract_usage(body) == Usage(input_tokens=2, cache_read_tokens=0,
                                        cache_write_tokens=2926, output_tokens=1)


def test_anthropic_still_preferred_when_no_openai_keys():
    body = {"usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_creation_input_tokens": 10, "cache_read_input_tokens": 30}}
    assert extract_usage(body) == Usage(input_tokens=100, cache_read_tokens=30,
                                        cache_write_tokens=10, output_tokens=20)


def test_openai_choices_usage_fallback():
    body = {"choices": [{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}]}
    assert extract_usage(body) == Usage(input_tokens=10, output_tokens=5)


def test_anthropic_nonstream():
    body = {"usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_creation_input_tokens": 10, "cache_read_input_tokens": 30}}
    assert extract_usage(body) == Usage(
        input_tokens=100, cache_read_tokens=30, cache_write_tokens=10, output_tokens=20)


def test_no_usage_returns_empty():
    assert extract_usage({"choices": [{"message": {"content": "hi"}}]}) == Usage()


def test_scanner_openai_single_chunk():
    s = SseUsageScanner()
    s.feed(b'data: {"choices":[]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"cached_tokens":40}}\n\n'
           b'data: [DONE]\n\n')
    assert s.usage == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_scanner_split_across_chunks():
    s = SseUsageScanner()
    payload = b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"cached_tokens":40}}\n\n'
    for i in range(0, len(payload), 7):
        s.feed(payload[i:i + 7])
    assert s.usage == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_scanner_anthropic_events():
    s = SseUsageScanner()
    s.feed(b'event: message_start\n'
           b'data: {"type":"message_start","message":{"usage":{"input_tokens":100,'
           b'"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"output_tokens":1}}}\n\n')
    s.feed(b'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n')
    s.feed(b'data: {"type":"message_delta","usage":{"output_tokens":20}}\n\n')
    assert s.usage == Usage(
        input_tokens=100, cache_read_tokens=30, cache_write_tokens=10, output_tokens=20)


def test_responses_nonstream_usage():
    # Responses API: input_tokens INCLUDES cached tokens; details carry the split.
    body = {"usage": {"input_tokens": 100, "output_tokens": 20,
                      "input_tokens_details": {"cached_tokens": 80}}}
    assert extract_usage(body) == Usage(input_tokens=20, cache_read_tokens=80,
                                        output_tokens=20)


def test_responses_nonstream_usage_without_details():
    body = {"usage": {"input_tokens": 100, "output_tokens": 20}}
    assert extract_usage(body) == Usage(input_tokens=100, output_tokens=20)


def test_responses_nonstream_usage_reasoning_only_output():
    # output_tokens_details.reasoning_tokens is part of output — no separate split
    body = {"usage": {"input_tokens": 50, "output_tokens": 30,
                      "output_tokens_details": {"reasoning_tokens": 25},
                      "input_tokens_details": {"cached_tokens": 10}}}
    assert extract_usage(body) == Usage(input_tokens=40, cache_read_tokens=10,
                                        output_tokens=30)


def test_scanner_responses_completed_event():
    s = SseUsageScanner()
    s.feed(b'data: {"type":"response.output_text.delta","delta":"H"}\n\n')
    s.feed(b'data: {"type":"response.completed","response":{"usage":'
           b'{"input_tokens":100,"output_tokens":20,'
           b'"input_tokens_details":{"cached_tokens":80}}}}\n\n')
    assert s.usage == Usage(input_tokens=20, cache_read_tokens=80, output_tokens=20)


def test_scanner_responses_created_event_has_no_usage():
    # protocol-opening event: must not wipe or pollute usage, not a content event
    s = SseUsageScanner()
    s.feed(b'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n')
    assert s.usage == Usage()
    assert s.has_content is False


def test_content_event_detection_responses():
    s = SseUsageScanner()
    s.feed(b'data: {"type":"response.created","response":{}}\n\n')
    assert s.has_content is False
    s.feed(b'data: {"type":"response.output_text.delta","delta":"H"}\n\n')
    assert s.has_content is True
    # tool-call argument deltas are generation output too
    s2 = SseUsageScanner()
    s2.feed(b'data: {"type":"response.function_call_arguments.delta","delta":"{"}\n\n')
    assert s2.has_content is True
    # completed is a protocol-closing event, not content
    s3 = SseUsageScanner()
    s3.feed(b'data: {"type":"response.completed","response":{}}\n\n')
    assert s3.has_content is False


def test_scanner_ignores_invalid_json_and_noise():
    s = SseUsageScanner()
    s.feed(b': keep-alive\n\n'
           b'event: ping\n'
           b'data: {not json}\n\n'
           b'data: [DONE]\n\n')
    assert s.usage == Usage()


def test_scanner_flush_captures_trailing_line_without_newline():
    s = SseUsageScanner()
    s.feed(b'data: {"choices":[]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"cached_tokens":40}}')
    assert s.usage == Usage()          # line not yet terminated
    s.flush()
    assert s.usage == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)
    s.flush()                          # idempotent
    assert s.usage == Usage(input_tokens=60, cache_read_tokens=40, output_tokens=20)


def test_content_event_detection_anthropic():
    s = SseUsageScanner()
    # message_start arrives before generation begins — not a content event
    s.feed(b'event: message_start\n'
           b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\n')
    assert s.has_content is False
    s.feed(b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')
    assert s.has_content is False
    s.feed(b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"h"}}\n\n')
    assert s.has_content is True


def test_content_event_detection_openai():
    s = SseUsageScanner()
    # opening frame with a role-only delta — not a content event
    s.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
    assert s.has_content is False
    # empty-content delta — not a content event
    s.feed(b'data: {"choices":[{"delta":{"content":""}}]}\n\n')
    assert s.has_content is False
    s.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    assert s.has_content is True


def test_content_event_detection_reasoning_and_tool():
    # thinking deltas and tool_use input count as generation output too
    s = SseUsageScanner()
    s.feed(b'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hm"}}\n\n')
    assert s.has_content is True
    s2 = SseUsageScanner()
    s2.feed(b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"}}\n\n')
    assert s2.has_content is True
    # OpenAI reasoning_content deltas count
    s3 = SseUsageScanner()
    s3.feed(b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n')
    assert s3.has_content is True
