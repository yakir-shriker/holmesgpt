from holmes.core.tool_search import TOOL_SEARCH_TOOL


def test_tool_search_tool_is_regex_variant():
    # Regex variant works on both the Anthropic API and Bedrock Invoke (BM25 does not).
    assert TOOL_SEARCH_TOOL["type"] == "tool_search_tool_regex_20251119"
    assert TOOL_SEARCH_TOOL["name"] == "tool_search_tool_regex"
