import logging
from typing import Any, Optional

from agents.prompt import coder_prompt

from .llm import ChatLLM, extract_code

logger = logging.getLogger(__name__)


class Coder:
    """Turns a plan into `execute_command(image)` against the ImagePatch API."""

    def __init__(
        self,
        coder_prompt: coder_prompt,
        llm: ChatLLM,
        model_name: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 2.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.coder_prompt = coder_prompt
        self.example_coder = coder_prompt.EXAMPLES_CODER
        self.plan = None
        self.thought = None
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        # Kept for API compatibility. Gemini's OpenAI-compat layer does not
        # honour these, so ChatLLM strips them per provider (see
        # `unsupported_params` in llm_config.yaml) rather than erroring.
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    async def __call__(self, plan: str, **kwargs) -> str:
        prompt = self.coder_prompt.CODE.format(plan=plan, example=self.example_coder)

        raw = await self.llm.chat(
            self.llm.system_prompt,
            prompt,
            agent="coder",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=['"""'],
        )
        code = extract_code(raw)

        if "def execute_command" not in code:
            logger.warning("Coder reply had no execute_command; re-prompting once")
            raw = await self.llm.chat(
                self.llm.system_prompt,
                prompt
                + "\n\nIMPORTANT: reply with ONLY a Python code block defining "
                  "`def execute_command(image):`. No prose, no explanation.",
                agent="coder",
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            code = extract_code(raw)

        return code
