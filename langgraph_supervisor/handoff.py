import re
import uuid
from typing import Optional, TypeGuard, cast

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from typing_extensions import Annotated

WHITESPACE_RE = re.compile(r"\s+")
METADATA_KEY_HANDOFF_DESTINATION = "__handoff_destination"
METADATA_KEY_IS_HANDOFF_BACK = "__is_handoff_back"


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name."""
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def _has_multiple_content_blocks(content: str | list[str | dict]) -> TypeGuard[list[dict]]:
    """Check if content contains multiple dict-based content blocks."""
    if not isinstance(content, list):
        return False
    dict_blocks = [block for block in content if isinstance(block, dict)]
    return len(dict_blocks) > 1


def _remove_non_handoff_tool_calls(
    last_ai_message: AIMessage, handoff_tool_call_id: str
) -> AIMessage:
    """Remove tool calls that are not meant for the agent."""
    content = last_ai_message.content
    if _has_multiple_content_blocks(content):
        filtered_content = []
        for content_block in content:
            if isinstance(content_block, str):
                filtered_content.append(content_block)
                continue

            is_target_tool_use = (
                content_block.get("type") == "tool_use"
                and content_block.get("id") == handoff_tool_call_id
            )
            is_non_tool_use = content_block.get("type") != "tool_use"
            if is_target_tool_use or is_non_tool_use:
                filtered_content.append(content_block)
        content = filtered_content

    return AIMessage(
        content=content,
        tool_calls=[
            tool_call
            for tool_call in last_ai_message.tool_calls
            if tool_call["id"] == handoff_tool_call_id
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )


def create_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None,
    add_handoff_messages: bool = True,
) -> BaseTool:
    """Create a tool that can handoff control to the requested agent."""
    if name is None:
        name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    if description is None:
        description = f"Ask agent '{agent_name}' for help"

    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name},
        )
        last_ai_message = cast(AIMessage, state["messages"][-1])

        if len(last_ai_message.tool_calls) > 1:
            handoff_messages = state["messages"][:-1]
            if add_handoff_messages:
                handoff_messages.extend(
                    (
                        _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                        tool_message,
                    )
                )
            return Command(
                graph=Command.PARENT,
                goto=[Send(agent_name, {**state, "messages": handoff_messages})],
            )

        if add_handoff_messages:
            handoff_messages = state["messages"] + [tool_message]
        else:
            handoff_messages = state["messages"][:-1]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={**state, "messages": handoff_messages},
        )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[AIMessage, ToolMessage]:
    """Create handoff-back messages for history stitching."""
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
            response_metadata={METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
    )


def create_forward_message_tool(supervisor_name: str = "supervisor") -> BaseTool:
    """Create a tool the supervisor can use to forward a worker message by name."""
    tool_name = "forward_message"
    desc = (
        "Forwards the latest message from the specified agent to the user"
        " without any changes. Use this to preserve information fidelity, avoid"
        " misinterpretation of questions or responses, and save time."
    )

    @tool(tool_name, description=desc)
    def forward_message(
        from_agent: str,
        state: Annotated[dict, InjectedState],
    ) -> str | Command:
        target_message = next(
            (
                m
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage)
                and (m.name or "").lower() == from_agent.lower()
                and not m.response_metadata.get(METADATA_KEY_IS_HANDOFF_BACK)
            ),
            None,
        )
        if not target_message:
            found_names = set(
                m.name for m in state["messages"] if isinstance(m, AIMessage) and m.name
            )
            return (
                f"Could not find message from source agent {from_agent}. Found names: {found_names}"
            )

        updates = [
            AIMessage(
                content=target_message.content,
                name=supervisor_name,
                id=str(uuid.uuid4()),
            )
        ]

        return Command(
            graph=Command.PARENT,
            goto="__end__",
            update={**state, "messages": updates},
        )

    return forward_message


def create_auto_forward_message_tool(
    *,
    supervisor_name: str = "toplevel-supervisor",
) -> BaseTool:
    """
    auto_forward_message tool:
      - サブagentの最終自然文AIMessageを拾って、supervisorの最終出力としてそのまま転送
      - さらに「tool_call と tool_result」を親グラフの履歴(messages)に残す
        -> 次ターンで LLM が前例を見て forward_message を再実行しやすくなる

    前提:
      - 親Stateの messages が `Annotated[list[AnyMessage], operator.add]` 等で
        append(reducer add) になってること

    旧ツールとの互換性:
      - ツール名は既存 `create_forward_message_tool` と同じ `forward_message`
      - 引数 `from_agent` + `InjectedState` を受ける呼び出し互換を維持
      - 見つからない場合はエラー文字列を返す挙動も維持
      - 差分は「tool_call/tool_result を合成して履歴に残す」点のみ
    """
    tool_name = "forward_message"

    def _is_effectively_empty_ai(message: AIMessage) -> bool:
        """Ignore AI messages that are effectively empty."""
        if getattr(message, "tool_calls", None):
            return False

        content = message.content
        if content is None:
            return True
        if isinstance(content, str):
            return content.strip() == ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        return False
                elif str(item).strip():
                    return False
            return True

        return str(content).strip() == ""

    def _is_handoff_back_marker(message: AIMessage) -> bool:
        metadata = getattr(message, "response_metadata", None) or {}
        return bool(isinstance(metadata, dict) and metadata.get(METADATA_KEY_IS_HANDOFF_BACK) is True)

    def _has_tool_calls(message: AIMessage) -> bool:
        return bool(getattr(message, "tool_calls", None) or [])

    @tool(tool_name)
    def forward_message(
        from_agent: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None,
    ) -> str | Command:
        """
        SHOW_RAW_SUBAGENT_OUTPUTS

        Purpose:
        - Render and show the raw output produced by a sub-agent to the user.

        HARD RULES (MUST FOLLOW):
        1) Trigger: If a sub-agent has finished and returned control, invoke this tool next.
        2) Do not emit user-facing text before invoking this tool.
        3) After invoking this tool, terminate the response immediately.

        Self-check (MANDATORY):
        - Before any user-visible response, check unrendered sub-agent output first.
        """
        # from_agent の最終自然文AIMessageを探して、その内容を supervisor の最終出力として転送。
        # その上で、親の履歴に
        #   1) tool_call相当AIMessage（合成）
        #   2) tool_result ToolMessage（合成）
        #   3) 最終出力AIMessage（転送本文）
        # を append する。
        state_messages = list(state.get("messages", []))

        # --- (A) 今回の forward_message の tool_call_id を取る
        # ★今回の“本物”の tool_call_id を使う(取れない時だけ保険でuuid)
        forward_call_id = tool_call_id or str(uuid.uuid4())

        # --- (B) 転送するターゲット（サブagentの最終自然文）を探す
        # ・AIMessage
        # ・name が from_agent と一致
        # ・tool_calls無し（=自然文）
        # ・handoff_back自動文じゃない
        # ・空AIじゃない
        target_message: Optional[AIMessage] = None
        for message in reversed(state_messages):
            if not isinstance(message, AIMessage):
                continue
            if (message.name or "").lower() != (from_agent or "").lower():
                continue
            if _is_handoff_back_marker(message):
                continue
            if _has_tool_calls(message):
                continue
            if _is_effectively_empty_ai(message):
                continue
            target_message = message
            break

        # --- (C) 親に append する messages を組み立て
        # ★ 1) tool_call相当AIMessageを「合成」して親履歴に残す
        #    これが無いと「tool_resultだけ残って、LLMが前例を見れない」問題が起きる
        messages_to_append = [
            AIMessage(
                content="",
                name=supervisor_name,
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"from_agent": from_agent},
                        "id": forward_call_id,
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={
                    "message_agent_name": supervisor_name,
                    "__synthetic_tool_call": True,
                },
            )
        ]

        if not target_message:
            found_names = set(
                m.name for m in state["messages"] if isinstance(m, AIMessage) and m.name
            )
            return (
                f"ERROR: Could not find message from source agent {from_agent}. "
                f"Found names: {found_names}"
            )
            # ★ 2) 失敗ToolMessageも残す（次ターンのデバッグに効く）
            # messages_to_append.append(
            #     ToolMessage(
            #         name=tool_name,
            #         tool_call_id=forward_call_id,
            #         content=f"ERROR: Could not find message from {from_agent}",
            #         artifact={
            #             "status": "error",
            #             "from_agent": from_agent,
            #             "message_agent_name": supervisor_name,
            #         },
            #     )
            # )
            #
            # ★ 3) 最終AI（ユーザに見せるなら）
            # messages_to_append.append(
            #     AIMessage(
            #         content=f"Could not find message from {from_agent}.",
            #         name=supervisor_name,
            #         id=str(uuid.uuid4()),
            #         additional_kwargs={"message_agent_name": supervisor_name},
            #     )
            # )
            #
            # 強制graph終了
            # return Command(
            #     graph=Command.PARENT,
            #     goto="__end__",
            #     # ✅ 上書きじゃなくて append だけ
            #     update={"messages": messages_to_append},
            # )

        # ★ 2) 成功ToolMessageを「合成」して親履歴に残す（tool_call_idで紐づけ）
        messages_to_append.append(
            ToolMessage(
                name=tool_name,
                tool_call_id=forward_call_id,
                content="Successfully forward_message, OK",
                artifact={
                    "status": "success",
                    "from_agent": from_agent,
                    "forwarded_message_id": getattr(target_message, "id", None),
                    "message_agent_name": supervisor_name,
                },
            )
        )

        # ★ 3) supervisor最終AI（転送本文）
        messages_to_append.append(
            AIMessage(
                content=target_message.content,
                name=supervisor_name,
                id=str(uuid.uuid4()),
                additional_kwargs={"message_agent_name": supervisor_name},
            )
        )

        return Command(
            graph=Command.PARENT,
            goto="__end__",
            # ✅ ここが大事：messages を “置換” しないで append する
            update={"messages": messages_to_append},
        )

    return forward_message


__all__ = [
    "METADATA_KEY_HANDOFF_DESTINATION",
    "METADATA_KEY_IS_HANDOFF_BACK",
    "create_auto_forward_message_tool",
    "create_forward_message_tool",
    "create_handoff_back_messages",
    "create_handoff_tool",
]
