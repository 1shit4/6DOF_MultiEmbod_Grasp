import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

from agents.prompt import observer_prompt

from .llm import ChatLLM, extract_json, extract_tag

logger = logging.getLogger(__name__)


def encode_image(image_path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Observer:
    """Vision critic: looks at the rendered grasp and judges it against the query.

    This is the closed-loop feedback that makes GraspMAS more than a one-shot
    pipeline, and it is why the LLM provider has to support image input.
    """

    def __init__(
        self,
        observer_prompt: observer_prompt,
        llm: ChatLLM,
        model_name: Optional[str] = None,
        max_tokens: int = 1000,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.observer_prompt = observer_prompt
        self.llm = llm

    async def __call__(self, execute_results: dict, query: str) -> dict:
        try:
            results_str = json.dumps(execute_results, indent=2, default=str)
        except Exception:
            results_str = str(execute_results)

        img_path = execute_results.get("image")
        prompt = self.observer_prompt.OBSERVER.format(
            results=results_str, user_query=query
        )

        # Every grasp path must emit a PNG; if one didn't, fall back to a
        # text-only critique rather than crashing the round.
        b64 = None
        if img_path and Path(str(img_path)).exists():
            b64 = encode_image(img_path)
        else:
            logger.warning("Observer got no usable image (%s); text-only critique", img_path)

        raw = await self._ask(prompt, b64)
        parsed = self._parse(raw)

        if parsed is None:
            logger.warning("Observer reply was not parseable JSON; re-prompting once")
            raw = await self._ask(
                prompt
                + "\n\nIMPORTANT: reply with ONLY a JSON object inside "
                  "<observation></observation> tags. No prose outside the tags.",
                b64,
            )
            parsed = self._parse(raw)

        if parsed is None:
            # Keep the loop alive with the raw text as the summary; the Planner
            # consumes `summary` as free text anyway.
            logger.warning("Observer still unparseable; passing raw text through")
            parsed = {"summary": (raw or "").strip()[:2000], "parse_error": True}

        parsed.setdefault("summary", "")
        return parsed

    async def _ask(self, prompt: str, b64: Optional[str]) -> str:
        if b64 is None:
            return await self.llm.chat(
                self.llm.system_prompt, prompt, agent="observer",
                max_tokens=self.max_tokens,
            )
        return await self.llm.chat_with_image(
            self.llm.system_prompt, prompt, b64, agent="observer",
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _parse(raw: str) -> Optional[dict]:
        inner = extract_tag(raw, "observation")
        for candidate in (inner, raw):
            if candidate:
                parsed = extract_json(candidate)
                if parsed is not None:
                    return parsed
        # Tagged, but the body was prose rather than JSON — still usable.
        if inner:
            return {"summary": inner}
        return None
