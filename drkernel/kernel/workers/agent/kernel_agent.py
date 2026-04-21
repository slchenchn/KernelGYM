import re

from kernel.utils.kernel_code import extract_kernel_submission
from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class KernelAgent(BaseAgent):
    """
    Agent that supports multi-turn code generation, capable of handling code execution, self-test, and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.answer_block_re = re.compile(
            r"""
            (?P<block>
                ```answer[ \t]*(?:\r?\n)?         # Opening fence: ```answer
                (?P<code>.*?)                     # Answer content
                (?:\r?\n)?```                     # Closing fence
            )
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

    async def generate_thought_and_action(
        self, response_token_ids: list[int], response_truncation: str
    ) -> tuple[str | None, str | None, bool | None, dict]:
        # remove padding token ids
        response_token_ids = [id for id in response_token_ids if id != self.tokenizer.pad_token_id]
        # translate result_token_id back to string
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)

        if response is None:
            return None, None, None, True, {}

        # Always treat any extracted block as final answer; otherwise no tool call
        answer_block = self._extract_answer_block(response)
        if answer_block is not None:
            return response, response_token_ids, answer_block, True, {
                'finish_type': FinishReasonTypeEnum.ANSWER
            }

        kernel_submission = self._extract_kernel_submission(response)
        if kernel_submission is not None:
            if kernel_submission.strip().startswith("### CUDA_KERNELS"):
                code_block = kernel_submission.strip()
            else:
                code_block = f"```python\n{kernel_submission.strip()}\n```"
            return response, response_token_ids, code_block, True, {
                'finish_type': FinishReasonTypeEnum.ANSWER
            }

        return response, response_token_ids, None, True, {
            'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL
        }

    def _extract_answer_block(self, response: str) -> str | None:
        match = self.answer_block_re.search(response)
        if match:
            return match.group("block")
        return None

    def _extract_kernel_submission(self, response: str) -> str | None:
        extracted = extract_kernel_submission(response, kernel_backend="cuda_agent")
        if extracted and extracted != response:
            return extracted

        extracted = extract_kernel_submission(response, kernel_backend="triton")
        if extracted and extracted.strip():
            return extracted
        return None
