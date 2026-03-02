import inspect
from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional, Type, Union, cast, get_args
from uuid import UUID, uuid5

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig, RunnableSequence
from langchain_core.tools import BaseTool
from langgraph._internal._config import patch_configurable
from langgraph._internal._runnable import RunnableCallable, RunnableLike
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.pregel import Pregel
from langgraph.pregel.remote import RemoteGraph
from langgraph.runtime import Runtime
from typing_extensions import Annotated, TypedDict

from langgraph_supervisor.agent_name import AgentNameMode, with_agent_name
from langgraph_supervisor.handoff import (
    METADATA_KEY_HANDOFF_DESTINATION,
    _normalize_agent_name,
    create_handoff_back_messages,
    create_handoff_tool,
)

OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow.

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""

MODELS_NO_PARALLEL_TOOL_CALLS = {"o3-mini", "o3", "o4-mini"}
PROMPT_RUNNABLE_NAME = "Prompt"

StructuredResponseSchema = dict[str, Any] | type[Any]
Prompt = (
    SystemMessage
    | str
    | Callable[[dict[str, Any]], Any]
    | Runnable[dict[str, Any], Any]
)


class _SupervisorAgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


class _SupervisorAgentStateWithStructuredResponse(_SupervisorAgentState, total=False):
    structured_response: Any


class _OuterState(TypedDict):
    """The state of the supervisor workflow."""

    messages: Annotated[Sequence[AnyMessage], add_messages]


class _OuterStateWithStructuredResponse(_OuterState, total=False):
    structured_response: Any


def _supports_disable_parallel_tool_calls(model: LanguageModelLike) -> bool:
    if not isinstance(model, BaseChatModel):
        return False

    if (
        model_name := getattr(model, "model_name", None)
    ) and model_name in MODELS_NO_PARALLEL_TOOL_CALLS:
        return False

    if not hasattr(model, "bind_tools"):
        return False

    if "parallel_tool_calls" not in inspect.signature(model.bind_tools).parameters:
        return False

    return True


def _get_state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _get_prompt_runnable(prompt: Prompt | None) -> Runnable:
    if prompt is None:
        return RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    if isinstance(prompt, str):
        system_message = SystemMessage(content=prompt)
        return RunnableCallable(
            lambda state: [system_message] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    if isinstance(prompt, SystemMessage):
        return RunnableCallable(
            lambda state: [prompt] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    if inspect.iscoroutinefunction(prompt):
        return RunnableCallable(None, prompt, name=PROMPT_RUNNABLE_NAME)
    if callable(prompt):
        return RunnableCallable(prompt, name=PROMPT_RUNNABLE_NAME)
    if isinstance(prompt, Runnable):
        return prompt

    raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")


def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in model.bind_tools() and tools passed to create_supervisor must match. "
            f"Got {len(tools)} tools, expected {len(bound_tools)}"
        )

    tool_names = {tool.name for tool in tools}
    bound_tool_names: set[str] = set()
    for bound_tool in bound_tools:
        if not isinstance(bound_tool, dict):
            continue
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            continue
        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in model.bind_tools()")

    return False


def _get_model(model: LanguageModelLike) -> BaseChatModel:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            "Expected `model` to be a chat model or RunnableBinding, "
            f"got {type(model)}"
        )

    return model


def _validate_chat_history(messages: Sequence[BaseMessage]) -> None:
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if tool_calls_without_results:
        raise ValueError(
            "Found AI tool_calls without corresponding ToolMessage. "
            f"First missing tool calls: {tool_calls_without_results[:3]}"
        )


def _make_call_agent(
    agent: Pregel[Any],
    output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
) -> RunnableCallable:
    if output_mode not in get_args(OutputMode):
        raise ValueError(
            f"Invalid agent output mode: {output_mode}. Needs to be one of {get_args(OutputMode)}"
        )

    def _process_output(output: dict[str, Any]) -> dict[str, Any]:
        messages = output["messages"]
        if output_mode == "last_message":
            if isinstance(messages[-1], ToolMessage):
                messages = messages[-2:]
            else:
                messages = messages[-1:]

        if add_handoff_back_messages:
            messages.extend(create_handoff_back_messages(agent.name, supervisor_name))

        return {
            **output,
            "messages": messages,
        }

    def call_agent(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        thread_id = config.get("configurable", {}).get("thread_id")
        output = agent.invoke(
            state,
            patch_configurable(
                config,
                {
                    "thread_id": str(uuid5(UUID(str(thread_id)), agent.name))
                    if thread_id
                    else None
                },
            )
            if isinstance(agent, RemoteGraph)
            else config,
        )
        return _process_output(output)

    async def acall_agent(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        thread_id = config.get("configurable", {}).get("thread_id")
        output = await agent.ainvoke(
            state,
            patch_configurable(
                config,
                {
                    "thread_id": str(uuid5(UUID(str(thread_id)), agent.name))
                    if thread_id
                    else None
                },
            )
            if isinstance(agent, RemoteGraph)
            else config,
        )
        return _process_output(output)

    return RunnableCallable(call_agent, acall_agent)


def _get_handoff_destinations(tools: Sequence[BaseTool | Callable]) -> list[str]:
    return [
        tool.metadata[METADATA_KEY_HANDOFF_DESTINATION]
        for tool in tools
        if isinstance(tool, BaseTool)
        and tool.metadata is not None
        and METADATA_KEY_HANDOFF_DESTINATION in tool.metadata
    ]


def _prepare_tool_node(
    tools: list[BaseTool | Callable] | ToolNode | None,
    handoff_tool_prefix: Optional[str],
    add_handoff_messages: bool,
    agent_names: set[str],
) -> ToolNode:
    if isinstance(tools, ToolNode):
        input_tool_node = tools
        tool_classes = list(tools.tools_by_name.values())
    elif tools:
        input_tool_node = ToolNode(tools)
        tool_classes = list(input_tool_node.tools_by_name.values())
    else:
        input_tool_node = None
        tool_classes = []

    handoff_destinations = _get_handoff_destinations(tool_classes)
    if handoff_destinations:
        missing = set(agent_names) - set(handoff_destinations)
        if missing:
            raise ValueError(
                "When providing custom handoff tools, you must provide them for all subagents. "
                f"Missing handoff tools for agents '{missing}'."
            )
        return cast(ToolNode, input_tool_node)

    handoff_tools = [
        create_handoff_tool(
            agent_name=agent_name,
            name=(
                None
                if handoff_tool_prefix is None
                else f"{handoff_tool_prefix}{_normalize_agent_name(agent_name)}"
            ),
            add_handoff_messages=add_handoff_messages,
        )
        for agent_name in agent_names
    ]

    all_tools = tool_classes + list(handoff_tools)
    if input_tool_node is not None:
        return ToolNode(
            all_tools,
            name=str(input_tool_node.name),
            tags=list(input_tool_node.tags) if input_tool_node.tags else None,
            handle_tool_errors=input_tool_node._handle_tool_errors,
            messages_key=input_tool_node._messages_key,
        )

    return ToolNode(all_tools)


def _build_supervisor_agent(
    *,
    supervisor_name: str,
    model: LanguageModelLike,
    tool_node: ToolNode,
    prompt: Prompt | None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ],
    pre_model_hook: Optional[RunnableLike],
    post_model_hook: Optional[RunnableLike],
    parallel_tool_calls: bool,
    include_agent_name: AgentNameMode | None,
    state_schema: Type[Any],
    context_schema: Type[Any] | None,
) -> Pregel:
    all_tools = list(tool_node.tools_by_name.values())
    should_return_direct = {t.name for t in all_tools if t.return_direct}

    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        model = cast(BaseChatModel, init_chat_model(model))

    if _should_bind_tools(model, all_tools) and all_tools:
        if _supports_disable_parallel_tool_calls(model):
            model = cast(BaseChatModel, model).bind_tools(
                all_tools, parallel_tool_calls=parallel_tool_calls
            )
        else:
            model = cast(BaseChatModel, model).bind_tools(all_tools)

    if include_agent_name:
        model = with_agent_name(model, include_agent_name)

    model_runnable = _get_prompt_runnable(prompt) | model

    def _get_model_input_state(state: dict[str, Any]) -> dict[str, Any]:
        if pre_model_hook is not None:
            messages = state.get("llm_input_messages") or state.get("messages")
        else:
            messages = state.get("messages")

        if messages is None:
            raise ValueError(
                "Expected input to model node to have 'messages' "
                "or 'llm_input_messages' (when pre_model_hook is used)."
            )

        _validate_chat_history(messages)
        state["messages"] = messages
        return state

    def call_model(
        state: dict[str, Any], runtime: Runtime[Any], config: RunnableConfig
    ) -> dict[str, Any]:
        _ = runtime
        model_input = _get_model_input_state(state)
        response = cast(AIMessage, model_runnable.invoke(model_input, config))
        response.name = supervisor_name
        return {"messages": [response]}

    async def acall_model(
        state: dict[str, Any], runtime: Runtime[Any], config: RunnableConfig
    ) -> dict[str, Any]:
        _ = runtime
        model_input = _get_model_input_state(state)
        response = cast(AIMessage, await model_runnable.ainvoke(model_input, config))
        response.name = supervisor_name
        return {"messages": [response]}

    def generate_structured_response(
        state: dict[str, Any], runtime: Runtime[Any], config: RunnableConfig
    ) -> dict[str, Any]:
        _ = runtime
        messages = list(_get_state_value(state, "messages", []))
        schema: StructuredResponseSchema | Any = response_format
        if isinstance(response_format, tuple):
            system_prompt, schema = response_format
            messages = [SystemMessage(content=system_prompt), *messages]

        structured = _get_model(model).with_structured_output(schema).invoke(messages, config)
        return {"structured_response": structured}

    async def agenerate_structured_response(
        state: dict[str, Any], runtime: Runtime[Any], config: RunnableConfig
    ) -> dict[str, Any]:
        _ = runtime
        messages = list(_get_state_value(state, "messages", []))
        schema: StructuredResponseSchema | Any = response_format
        if isinstance(response_format, tuple):
            system_prompt, schema = response_format
            messages = [SystemMessage(content=system_prompt), *messages]

        structured = await _get_model(model).with_structured_output(schema).ainvoke(
            messages, config
        )
        return {"structured_response": structured}

    def should_continue(state: dict[str, Any]) -> str:
        messages = _get_state_value(state, "messages", [])
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            if post_model_hook is not None:
                return "post_model_hook"
            if response_format is not None:
                return "generate_structured_response"
            return END

        if post_model_hook is not None:
            return "post_model_hook"
        return "tools"

    workflow = StateGraph(state_schema=state_schema, context_schema=context_schema)
    workflow.add_node("model", RunnableCallable(call_model, acall_model))
    workflow.add_node("tools", tool_node)

    if pre_model_hook is not None:
        workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
        workflow.add_edge("pre_model_hook", "model")
        entrypoint = "pre_model_hook"
    else:
        entrypoint = "model"

    workflow.set_entry_point(entrypoint)

    model_paths: list[str] = []
    post_hook_paths: list[str] = [entrypoint, "tools"]

    if post_model_hook is not None:
        workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
        workflow.add_edge("model", "post_model_hook")
        model_paths.append("post_model_hook")
    else:
        model_paths.append("tools")

    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(
                generate_structured_response,
                agenerate_structured_response,
            ),
        )
        if post_model_hook is not None:
            post_hook_paths.append("generate_structured_response")
        else:
            model_paths.append("generate_structured_response")
    else:
        if post_model_hook is not None:
            post_hook_paths.append(END)
        else:
            model_paths.append(END)

    if post_model_hook is not None:

        def post_model_hook_router(state: dict[str, Any]) -> str:
            messages = _get_state_value(state, "messages", [])
            if not messages:
                return END

            tool_messages = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
            last_ai_message = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)),
                None,
            )
            if last_ai_message is None:
                return END

            pending_tool_calls = [
                c for c in last_ai_message.tool_calls if c["id"] not in tool_messages
            ]
            if pending_tool_calls:
                return "tools"
            if isinstance(messages[-1], ToolMessage):
                return entrypoint
            if response_format is not None:
                return "generate_structured_response"
            return END

        workflow.add_conditional_edges(
            "post_model_hook",
            post_model_hook_router,
            path_map=post_hook_paths,
        )

    workflow.add_conditional_edges("model", should_continue, path_map=model_paths)

    def route_tool_responses(state: dict[str, Any]) -> str:
        last_non_tool: BaseMessage | None = None
        for message in reversed(_get_state_value(state, "messages", [])):
            if not isinstance(message, ToolMessage):
                last_non_tool = message
                break
            if message.name in should_return_direct:
                return END

        if isinstance(last_non_tool, AIMessage) and last_non_tool.tool_calls:
            if any(call["name"] in should_return_direct for call in last_non_tool.tool_calls):
                return END

        return entrypoint

    if should_return_direct:
        workflow.add_conditional_edges(
            "tools", route_tool_responses, path_map=[entrypoint, END]
        )
    else:
        workflow.add_edge("tools", entrypoint)

    return workflow.compile(name=supervisor_name)


def _resolve_forward_tool(
    tools_arg: list[BaseTool | Callable] | ToolNode | None,
    explicit_forward_tool: BaseTool | None,
) -> BaseTool | None:
    if explicit_forward_tool is not None:
        return explicit_forward_tool
    if tools_arg is None:
        return None

    if isinstance(tools_arg, ToolNode):
        return tools_arg.tools_by_name.get("forward_message")

    for tool_obj in tools_arg:
        if isinstance(tool_obj, BaseTool) and getattr(tool_obj, "name", None) == "forward_message":
            return tool_obj

    return None


def _invoke_forward_tool_direct(
    forward_tool: BaseTool,
    from_agent: str,
    state: dict[str, Any],
) -> Any:
    func = getattr(forward_tool, "func", None)
    if callable(func):
        return func(from_agent=from_agent, state=state)

    return forward_tool.invoke({"from_agent": from_agent, "state": state})


def create_supervisor(
    agents: list[Pregel],
    *,
    model: LanguageModelLike,
    tools: list[BaseTool | Callable] | ToolNode | None = None,
    prompt: Prompt | None = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    parallel_tool_calls: bool = False,
    state_schema: Type[Any] | None = None,
    context_schema: Type[Any] | None = None,
    output_mode: OutputMode = "last_message",
    add_handoff_messages: bool = True,
    handoff_tool_prefix: Optional[str] = None,
    add_handoff_back_messages: Optional[bool] = None,
    supervisor_name: str = "supervisor",
    include_agent_name: AgentNameMode | None = None,
    force_forward_agents: Optional[set[str]] = None,
    forward_tool: Optional[BaseTool] = None,
    **deprecated_kwargs: Any,
) -> StateGraph:
    """Create a multi-agent supervisor without create_react_agent dependency.

    Additional kwargs:
    - force_forward_agents: set of worker names to auto-forward and terminate.
    - forward_tool: explicit forward_message tool. If omitted, auto-detected from `tools`.

    Per-agent override:
    - agent.force_forward = True
    """
    if (config_schema := deprecated_kwargs.get("config_schema")) is not None:
        context_schema = config_schema

    if add_handoff_back_messages is None:
        add_handoff_back_messages = add_handoff_messages

    supervisor_schema = state_schema or (
        _SupervisorAgentStateWithStructuredResponse
        if response_format is not None
        else _SupervisorAgentState
    )
    workflow_schema = state_schema or (
        _OuterStateWithStructuredResponse if response_format is not None else _OuterState
    )

    agent_names: set[str] = set()
    for agent in agents:
        if agent.name is None or agent.name == "LangGraph":
            raise ValueError(
                "Please specify a name when creating sub-agents "
                "(e.g. graph.compile(name='your_agent'))."
            )
        if agent.name in agent_names:
            raise ValueError(
                f"Agent with name '{agent.name}' already exists. Agent names must be unique."
            )
        agent_names.add(agent.name)

    tool_node = _prepare_tool_node(
        tools,
        handoff_tool_prefix,
        add_handoff_messages,
        agent_names,
    )

    supervisor_agent = _build_supervisor_agent(
        supervisor_name=supervisor_name,
        model=model,
        tool_node=tool_node,
        prompt=prompt,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        parallel_tool_calls=parallel_tool_calls,
        include_agent_name=include_agent_name,
        state_schema=cast(Type[Any], supervisor_schema),
        context_schema=context_schema,
    )

    force_set = set(force_forward_agents or set())
    any_agent_flag = any(bool(getattr(a, "force_forward", False)) for a in agents)

    resolved_forward_tool: BaseTool | None = None
    if force_set or any_agent_flag:
        resolved_forward_tool = _resolve_forward_tool(tools, forward_tool)
        if resolved_forward_tool is None:
            raise ValueError(
                "Using force_forward_agents or agent.force_forward requires `forward_tool` "
                "or a tool named 'forward_message' in `tools`."
            )

    builder = StateGraph(cast(Type[Any], workflow_schema), context_schema=context_schema)
    builder.add_node(supervisor_agent, destinations=tuple(agent_names) + (END,))
    builder.add_edge(START, supervisor_agent.name)

    def _should_force(agent_obj: Pregel[Any]) -> bool:
        if bool(getattr(agent_obj, "force_forward", False)):
            return True
        if agent_obj.name in force_set:
            return True
        return False

    for agent in agents:
        builder.add_node(
            agent.name,
            _make_call_agent(
                agent,
                output_mode,
                add_handoff_back_messages=bool(add_handoff_back_messages),
                supervisor_name=supervisor_name,
            ),
        )

        if _should_force(agent):
            auto_node_name = f"auto_forward__{agent.name}"

            def _auto_forward_node(
                state: dict[str, Any],
                *args: Any,
                _agent_name: str = agent.name,
                **kwargs: Any,
            ) -> dict[str, Any]:
                _ = args, kwargs
                res = _invoke_forward_tool_direct(
                    cast(BaseTool, resolved_forward_tool),
                    from_agent=_agent_name,
                    state=state,
                )
                update = getattr(res, "update", None) or {}
                return {"messages": update.get("messages", [])}

            builder.add_node(auto_node_name, _auto_forward_node)
            builder.add_edge(agent.name, auto_node_name)
            builder.add_edge(auto_node_name, END)
        else:
            builder.add_edge(agent.name, supervisor_agent.name)

    return builder


__all__ = ["create_supervisor", "OutputMode"]
