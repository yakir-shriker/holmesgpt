"""Anthropic Tool Search (advanced tool use) support.

When ``HOLMES_TOOL_SEARCH_ENABLED`` is set and the active model is an Anthropic
(Claude) model, heavy tool schemas are marked ``defer_loading: true`` and the regex
tool-search tool below is added to the request. The model then discovers and loads
those tool definitions on demand instead of paying for the entire tool catalog in
the system prompt on every turn — which on large MCP deployments can be 10-60k tokens
before the user even sends a message. Verified ~89% prompt-token reduction on a
``hi`` turn (Claude Opus 4.5, 30 fabricated tools).

LiteLLM manages the required ``anthropic-beta`` header automatically once it sees the
tool-search tool, mapping it to the provider-specific value (e.g. Bedrock uses
``tool-search-tool-2025-10-19``), so we deliberately do NOT set the header ourselves —
doing so breaks Bedrock with "invalid beta flag".

Note: tool search needs an API that supports it — the Anthropic Messages API (direct
or via an Anthropic-compatible gateway) or the Bedrock *Invoke* route
(``bedrock/invoke/...``). The Bedrock *Converse* API does not support it.

Refs:
- https://docs.litellm.ai/docs/providers/anthropic_tool_search
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
"""

# Regex variant works across the Anthropic API and Bedrock Invoke; the BM25 variant
# is unsupported on Bedrock, so regex is the safe cross-provider default.
TOOL_SEARCH_TOOL: dict = {
    "type": "tool_search_tool_regex_20251119",
    "name": "tool_search_tool_regex",
}
