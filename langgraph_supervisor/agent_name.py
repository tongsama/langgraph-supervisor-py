import json
import re
from typing import Literal, Optional, Sequence, TypeGuard, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    MessageLikeRepresentation,
    convert_to_messages,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

NAME_PATTERN = re.compile(r"<name>(.*?)</name>", re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.DOTALL)

AgentNameMode = Literal["inline", "inline_xml", "inline_yaml", "inline_json"]


def _is_content_blocks_content(content: list[dict | str] | str) -> TypeGuard[list[dict | str]]:
    return isinstance(content, list)


def _split_text_and_non_text_blocks(content_list: list[dict | str]) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    non_text: list[dict] = []
    for block in content_list:
        if isinstance(block, str):
            text_parts.append(block)
            continue

        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        else:
            non_text.append(block)

    return "".join(text_parts), non_text


def add_inline_agent_name(message: BaseMessage) -> BaseMessage:
    """Add name and content XML tags to the message content."""
    if not isinstance(message, AIMessage) or not message.name:
        return message

    formatted_message = message.model_copy()
    if _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        formatted_content = f"<name>{message.name}</name><content>{text}</content>"
        formatted_message.content = [{"type": "text", "text": formatted_content}] + non_text_blocks
    else:
        formatted_message.content = (
            f"<name>{message.name}</name><content>{formatted_message.content}</content>"
        )
    return formatted_message


def remove_inline_agent_name(message: BaseMessage) -> BaseMessage:
    """Remove explicit name/content XML tags from AI message content."""
    if not isinstance(message, AIMessage) or not message.content:
        return message

    if is_content_blocks_content := _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        content_for_parse = text
    else:
        non_text_blocks = []
        content_for_parse = str(message.content)

    name_match = NAME_PATTERN.search(content_for_parse)
    content_match = CONTENT_PATTERN.search(content_for_parse)
    if not name_match or not content_match:
        return message

    parsed_content = content_match.group(1)
    parsed_message = message.model_copy()

    if is_content_blocks_content:
        content_blocks = non_text_blocks
        if parsed_content:
            content_blocks = [{"type": "text", "text": parsed_content}] + content_blocks
        parsed_message.content = cast(list[str | dict], content_blocks)
    else:
        parsed_message.content = parsed_content

    return parsed_message


def _to_inline_yaml(name: str, content: str) -> str:
    name_yaml = json.dumps(name, ensure_ascii=False)
    raw_lines = str(content).split("\n")
    indented = "\n".join("  " + line for line in raw_lines)
    return (
        "__inline_agent_name: true\n"
        f"name: {name_yaml}\n"
        "content: |-\n"
        f"{indented}\n"
    )


def _try_parse_inline_yaml(s: str) -> Optional[tuple[Optional[str], str]]:
    if not isinstance(s, str):
        return None
    t = s.lstrip()
    if not t.startswith("__inline_agent_name: true"):
        return None

    lines = t.split("\n")
    inline_flag_ok = False
    parsed_name: Optional[str] = None
    parsed_content: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip() == "__inline_agent_name: true":
            inline_flag_ok = True

        if line.startswith("name:"):
            v = line[len("name:") :].strip()
            if v.startswith('"'):
                try:
                    parsed_name = json.loads(v)
                except Exception:
                    parsed_name = v.strip('"')
            else:
                parsed_name = v

        if line.startswith("content:"):
            v = line[len("content:") :].strip()
            if v.startswith("|"):
                i += 1
                block_lines: list[str] = []
                while i < len(lines):
                    l2 = lines[i]
                    if l2 == "" and i == len(lines) - 1:
                        break
                    if l2.startswith("  "):
                        block_lines.append(l2[2:])
                        i += 1
                        continue
                    return None
                parsed_content = "\n".join(block_lines)
            else:
                if v.startswith('"'):
                    try:
                        parsed_content = json.loads(v)
                    except Exception:
                        parsed_content = v.strip('"')
                else:
                    parsed_content = v

        i += 1

    if not inline_flag_ok or parsed_content is None:
        return None
    return parsed_name, parsed_content


def _to_inline_json(name: str, content: str) -> str:
    obj = {
        "__inline_agent_name": True,
        "name": name,
        "content": str(content),
    }
    return json.dumps(obj, ensure_ascii=False)


def _try_parse_inline_json(s: str) -> Optional[tuple[Optional[str], str]]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None

    try:
        obj = json.loads(t)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("__inline_agent_name") is not True:
        return None
    if "content" not in obj:
        return None

    return obj.get("name"), str(obj.get("content", ""))


def _add_inline_agent_name_yaml(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not message.name:
        return message

    formatted_message = message.model_copy()
    if _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        formatted_message.content = [{"type": "text", "text": _to_inline_yaml(message.name, text)}] + non_text_blocks
    else:
        formatted_message.content = _to_inline_yaml(message.name, str(formatted_message.content))

    return formatted_message


def _remove_inline_agent_name_yaml(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not message.content:
        return message

    if _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        content_for_parse = text
    else:
        non_text_blocks = []
        content_for_parse = str(message.content)

    parsed = _try_parse_inline_yaml(content_for_parse)
    if not parsed:
        return message

    parsed_name, parsed_content = parsed
    parsed_message = message.model_copy()

    if parsed_name and not getattr(parsed_message, "name", None):
        parsed_message.name = parsed_name

    if _is_content_blocks_content(message.content):
        blocks = non_text_blocks
        if parsed_content:
            blocks = [{"type": "text", "text": parsed_content}] + blocks
        parsed_message.content = cast(list[str | dict], blocks)
    else:
        parsed_message.content = parsed_content

    return parsed_message


def _add_inline_agent_name_json(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not message.name:
        return message

    formatted_message = message.model_copy()
    if _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        formatted_message.content = [{"type": "text", "text": _to_inline_json(message.name, text)}] + non_text_blocks
    else:
        formatted_message.content = _to_inline_json(message.name, str(formatted_message.content))

    return formatted_message


def _remove_inline_agent_name_json(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not message.content:
        return message

    if _is_content_blocks_content(message.content):
        text, non_text_blocks = _split_text_and_non_text_blocks(message.content)
        content_for_parse = text
    else:
        non_text_blocks = []
        content_for_parse = str(message.content)

    parsed = _try_parse_inline_json(content_for_parse)
    if not parsed:
        return message

    parsed_name, parsed_content = parsed
    parsed_message = message.model_copy()

    if parsed_name and not getattr(parsed_message, "name", None):
        parsed_message.name = parsed_name

    if _is_content_blocks_content(message.content):
        blocks = non_text_blocks
        if parsed_content:
            blocks = [{"type": "text", "text": parsed_content}] + blocks
        parsed_message.content = cast(list[str | dict], blocks)
    else:
        parsed_message.content = parsed_content

    return parsed_message


def with_agent_name(
    model: LanguageModelLike,
    agent_name_mode: AgentNameMode,
) -> LanguageModelLike:
    """Attach formatted agent names to model input/output message streams."""
    mode = str(agent_name_mode)
    if mode in ("inline", "inline_xml"):
        process_input_message = add_inline_agent_name
        process_output_message = remove_inline_agent_name
    elif mode == "inline_yaml":
        process_input_message = _add_inline_agent_name_yaml
        process_output_message = _remove_inline_agent_name_yaml
    elif mode == "inline_json":
        process_input_message = _add_inline_agent_name_json
        process_output_message = _remove_inline_agent_name_json
    else:
        raise ValueError(
            f"Invalid agent name mode: {agent_name_mode}. Needs to be one of {AgentNameMode.__args__}"
        )

    def process_input_messages(
        input: Sequence[MessageLikeRepresentation] | PromptValue,
    ) -> list[BaseMessage]:
        messages = convert_to_messages(input)
        return [process_input_message(message) for message in messages]

    chain = (
        process_input_messages
        | model
        | RunnableLambda(process_output_message, name="process_output_message")
    )

    return cast(LanguageModelLike, chain)


__all__ = [
    "AgentNameMode",
    "add_inline_agent_name",
    "remove_inline_agent_name",
    "with_agent_name",
]
