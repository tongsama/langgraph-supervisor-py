# README_MIGRATION_NOTES

このファイルは `langgraph-supervisor-py` に対して、ローカルで実施した移行・拡張内容の要約です。

## 目的
- `create_react_agent` 依存を無くし、LangChain v1 / LangGraph v1 で使用できる supervisor 実装にする。

## 主要変更
1. `langgraph_supervisor/supervisor.py`
- Supervisor 本体を `StateGraph` 手組みベースへ移行。
- `tools` で `ToolNode` を受け取る互換性を維持。
- `create_supervisor` に `force_forward_agents` と `forward_tool` を追加。

2. `langgraph_supervisor/agent_name.py`
- `AgentNameMode` を拡張。
- 対応モード: `inline`, `inline_xml`, `inline_yaml`, `inline_json`。
- mixed content blocks（`list[dict | str]`）を安全に処理する実装を追加。

3. `langgraph_supervisor/handoff.py`
- 既存 `create_forward_message_tool` は維持。
- 新規 `create_auto_forward_message_tool` を追加。
- `forward_message` 実行時に、合成 `tool_call` / `tool_result` / 最終転送 `AIMessage` を履歴へ append できるようにした。

4. `langgraph_supervisor/__init__.py`
- `create_auto_forward_message_tool` を公開APIに追加。

5. `tests/test_supervisor.py`
- upstream テスト内の `create_react_agent` 呼び出しを `create_agent` に置換。
- `model.bind_tools(...)` 前提の記述を `create_agent` 側での tool bind 前提に合わせて調整。
- これにより、テスト実行時の `create_react_agent` 非推奨警告を回避。

## 互換性メモ
- 旧 `create_forward_message_tool` はそのまま使用可能。
- 新 `create_auto_forward_message_tool` もツール名は `forward_message` を使用。
- `from_agent` + `InjectedState` 呼び出し互換を維持。

## 追加引数（create_supervisor）
- `force_forward_agents: Optional[set[str]] = None`
- `forward_tool: Optional[BaseTool] = None`
