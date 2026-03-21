from __future__ import annotations

from typing import Any

from vllm.entrypoints.logger import RequestLogger

try:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ErrorResponse,
    )
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels

    _USES_SPLIT_OPENAI_MODULES = False
except ModuleNotFoundError:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

    _USES_SPLIT_OPENAI_MODULES = True

try:
    from vllm.worker.worker_base import WorkerWrapperBase

    _USES_LEGACY_WORKER_WRAPPER = True
except ModuleNotFoundError:
    try:
        # vLLM v1 exposes the Ray-side wrapper separately from the bare worker
        # base. Prefer the Ray wrapper first because it is the closest semantic
        # match to the legacy WorkerWrapperBase used by the rollout codepath.
        from vllm.v1.executor.ray_utils import RayWorkerWrapper as WorkerWrapperBase
    except ImportError:
        # Fall back to the v1 bare worker base only when the Ray wrapper entry
        # point is unavailable.
        from vllm.v1.worker.worker_base import WorkerWrapperBase

    _USES_LEGACY_WORKER_WRAPPER = False


def build_worker_wrapper(*, vllm_config: Any, rpc_rank: int = 0, global_rank: int | None = None):
    if _USES_LEGACY_WORKER_WRAPPER:
        return WorkerWrapperBase(vllm_config=vllm_config)
    return WorkerWrapperBase(rpc_rank=rpc_rank, global_rank=global_rank)


def build_openai_chat_serving(
    engine_client: Any,
    *,
    model_name: str,
    model_path: str,
    request_logger: RequestLogger | None,
    response_role: str = "assistant",
    chat_template: str | None = None,
    chat_template_content_format: str = "auto",
    enable_auto_tools: bool = False,
    tool_parser: str | None = None,
):
    base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

    if _USES_SPLIT_OPENAI_MODULES:
        models = OpenAIServingModels(
            engine_client=engine_client,
            base_model_paths=base_model_paths,
        )
        openai_serving_render = OpenAIServingRender(
            model_config=engine_client.model_config,
            renderer=engine_client.renderer,
            io_processor=engine_client.io_processor,
            model_registry=models.registry,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
            default_chat_template_kwargs={},
        )
        serving_chat = OpenAIServingChat(
            engine_client,
            models,
            response_role,
            openai_serving_render=openai_serving_render,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            default_chat_template_kwargs={},
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
        )
        return models, serving_chat

    model_config = engine_client.model_config
    models = OpenAIServingModels(engine_client, model_config, base_model_paths)
    serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        models,
        response_role,
        request_logger=request_logger,
        chat_template=chat_template,
        chat_template_content_format=chat_template_content_format,
        enable_auto_tools=enable_auto_tools,
        tool_parser=tool_parser,
    )
    return models, serving_chat


def get_error_response_code(error_response: ErrorResponse) -> int:
    code = getattr(error_response, "code", None)
    if code is not None:
        return code

    error = getattr(error_response, "error", None)
    nested_code = getattr(error, "code", None)
    if nested_code is not None:
        return nested_code

    return 500
