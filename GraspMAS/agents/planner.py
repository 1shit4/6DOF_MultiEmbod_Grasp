import logging
from typing import Any, Optional, Tuple

from agents.prompt import planner_prompt

from .llm import ChatLLM, extract_tag

logger = logging.getLogger(__name__)


class Planner:
    """Decides the next step toward a 6-DoF grasp for the referring expression."""

    def __init__(
        self,
        planner_prompt: planner_prompt,
        llm: ChatLLM,
        model_name: Optional[str] = None,
        max_tokens: int = 1000,
    ):
        self.model_name = model_name  # None = whatever the provider resolved
        self.max_tokens = max_tokens
        self.planner_prompt = planner_prompt
        self.example_planner = planner_prompt.EXAMPLES_PLANNER
        self.plan = None
        self.thought = None
        self.llm = llm

    async def __call__(
        self, query: str, previous_plan: str, observation: str, **kwargs
    ) -> Tuple[str, str]:
        prompt = self.planner_prompt.PLAN.format(
            user_query=query,
            examples=self.example_planner,
            planning=previous_plan,
            observer_output=observation,
        )

        raw = await self.llm.chat(
            self.llm.system_prompt,
            prompt,
            agent="planner",
            temperature=0.8,
            top_p=0.9,
            max_tokens=self.max_tokens,
            stop=['"""'],
        )
        thought, plan = self._parse(raw)

        if plan is None:
            # One corrective re-prompt before giving up: models drop the tags
            # occasionally, and losing a whole round to that is expensive on a
            # rate-limited free tier.
            logger.warning("Planner reply missing <plan> tags; re-prompting once")
            raw = await self.llm.chat(
                self.llm.system_prompt,
                prompt
                + "\n\nIMPORTANT: reply with exactly this structure and nothing "
                  "else:\n<thought>your reasoning</thought>\n<plan>the plan</plan>",
                agent="planner",
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
            thought, plan = self._parse(raw)

        if plan is None:
            # Degrade rather than crash — the whole reply is usually still a
            # usable plan, just untagged.
            thought = thought or ""
            plan = (raw or "").strip()

        self.thought, self.plan = thought, plan
        return thought, plan

    @staticmethod
    def _parse(raw: str) -> Tuple[Optional[str], Optional[str]]:
        return extract_tag(raw, "thought"), extract_tag(raw, "plan")
