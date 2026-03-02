## Background / 背景
This fork contains local migration work to make the supervisor implementation compatible with LangChain v1, plus practical extensions used in production-like workflows.

この fork には、LangChain v1 対応の移行と実運用に近い拡張がローカル実装されています。これらを fork 本体に取り込むための提案です。

## Scope / 対象変更
### 1) Supervisor core migration
- Removed dependency on `create_react_agent` in supervisor implementation.
- Preserved compatibility for tool wiring, including `ToolNode`-based flows.

### 2) API additions
- `create_supervisor(..., force_forward_agents: Optional[set[str]] = None, forward_tool: Optional[BaseTool] = None)`

### 3) Agent name mode extension
- Added support for: `inline`, `inline_xml`, `inline_yaml`, `inline_json`.

### 4) Mixed content blocks support
- Improved handling for mixed content blocks (`list[dict | str]`) in message content.
- Prevented failures caused by assuming every block is dict-like (e.g. direct `block["type"]` access).
- Stabilized inline name formatting/parsing paths under mixed provider outputs.

### 5) Handoff extension
- Added `create_auto_forward_message_tool(...)`.
- Kept `create_forward_message_tool(...)` for backward compatibility.
- Exported new API from package `__init__.py`.

### 6) Upstream test migration
- Migrated `tests/test_supervisor.py` from `create_react_agent` to `create_agent`.
- Reduced deprecation-warning dependency in test execution.

## Files changed / 変更ファイル
- `langgraph_supervisor/supervisor.py`
- `langgraph_supervisor/agent_name.py`
- `langgraph_supervisor/handoff.py`
- `langgraph_supervisor/__init__.py`
- `tests/test_supervisor.py`
- `README_MIGRATION_NOTES.md` (notes)

## Validation / 検証
- Library tests pass locally.
- Integration tests pass in sibling test directory.
- Real LLM smoke scenario passes.

## Discussion points / 相談ポイント
1. Keep both `create_forward_message_tool` and `create_auto_forward_message_tool` as public APIs?
2. Keep this as one PR or split into core/API/test PRs?
3. Any preferred policy for mixed content normalization (strict vs permissive)?
