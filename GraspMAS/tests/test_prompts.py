"""The prompt files are a contract with the LLM, so they get tested like one.

Two classes of bug live here and neither shows up until a live run:

1. **Template breakage.** All three prompts are `str.format()` templates, so a
   stray literal `{` or `}` raises KeyError at call time. Adding a JSON example
   to the coder prompt did exactly that during this port.
2. **Contract drift.** If a prompt advertises a method the code does not have,
   the Coder writes code that cannot run — and the failure surfaces as a
   confusing runtime error several agents downstream.
"""

import inspect
import re

import pytest

from agents.prompt import coder_prompt, observer_prompt, planner_prompt


class TestTemplatesFormat:
    """These would have caught the `{ "pose": ... }` KeyError immediately."""

    def test_coder_code_template_formats(self):
        out = coder_prompt.CODE.format(
            plan="a plan", example=coder_prompt.EXAMPLES_CODER
        )
        assert "a plan" in out

    def test_planner_plan_template_formats(self):
        out = planner_prompt.PLAN.format(
            user_query="grasp the mug",
            examples=planner_prompt.EXAMPLES_PLANNER,
            planning="prev",
            observer_output="obs",
        )
        assert "grasp the mug" in out

    def test_observer_template_formats(self):
        # `results` is JSON, so it is full of braces — the template must
        # tolerate that in the substituted value.
        out = observer_prompt.OBSERVER.format(
            results='{"grasp": {"score": 0.8}}', user_query="grasp the mug"
        )
        assert "grasp the mug" in out

    @pytest.mark.parametrize(
        "template,placeholders",
        [
            ("CODE", {"plan", "example"}),
            ("PLAN", {"user_query", "examples", "planning", "observer_output"}),
        ],
    )
    def test_only_expected_placeholders(self, template, placeholders):
        """Any other {...} in a template is an unescaped literal brace."""
        src = getattr(coder_prompt, template, None) or getattr(planner_prompt, template)
        found = set(re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", src))
        assert found <= placeholders, f"unexpected placeholders: {found - placeholders}"


class TestNoStale2DLanguage:
    """The whole point of the port is that grasps are no longer rectangles."""

    def test_coder_does_not_promise_a_rectangle(self):
        spec = coder_prompt.CODE
        block = spec[spec.index("def grasp_detection") : spec.index("def overlaps_with")]
        # The signature and Returns section must describe a dict, not the old
        # planar [quality, x, y, w, h, angle] list. (Prose that explicitly says
        # "not a 2D rectangle" is fine and intended.)
        assert "List[float]" not in block
        assert "->dict" in block.replace(" ", "")
        assert "a rectangle represented by" not in block
        assert "quality, x, y, w, h, angle" not in block

    def test_planner_does_not_say_grasp_rectangle(self):
        text = planner_prompt.PLAN.lower()
        assert "grasp pose rectangle" not in text
        assert "computes grasp rectangles" not in text

    def test_observer_describes_the_6dof_rendering(self):
        text = observer_prompt.OBSERVER
        assert "rectangle in yellow and purple" not in text
        assert "approach" in text.lower()

    def test_coder_documents_the_6dof_return_keys(self):
        spec = coder_prompt.CODE
        for key in ("position", "approach", "closing", "score", "width"):
            assert f'"{key}"' in spec, f"grasp_detection return key {key} undocumented"


class TestDepthSemantics:
    """compute_depth flipped from relative inverse depth to metres, so the
    prompts must say which direction 'closer' now is."""

    def test_coder_states_metres(self):
        assert "METRES" in coder_prompt.CODE or "metres" in coder_prompt.CODE

    def test_coder_states_smaller_is_closer(self):
        assert re.search(r"smaller\s+(means|is)\s+closer", coder_prompt.CODE, re.I)

    def test_planner_states_smaller_is_closer(self):
        assert re.search(r"smaller\s+means\s+closer", planner_prompt.PLAN, re.I)

    def test_compute_depth_signature_is_argument_free(self):
        """Upstream advertised `compute_depth(object_name: str)`, which the real
        method never accepted."""
        assert "compute_depth(object_name" not in planner_prompt.PLAN


class TestAdvertisedApiMatchesCode:
    def test_every_advertised_method_exists(self):
        from image_patch import ImagePatch  # noqa: F401  (heavy import)

        advertised = set(re.findall(r"def (\w+)\(", coder_prompt.CODE))
        # `__init__` and helpers documented for the LLM but not tool methods.
        advertised -= {"execute_command"}
        for name in advertised:
            assert hasattr(ImagePatch, name), f"prompt advertises missing method {name}"

    def test_grasp_detection_signature_matches(self):
        from image_patch import ImagePatch

        sig = inspect.signature(ImagePatch.grasp_detection)
        assert "object_patch" in sig.parameters
        assert "gripper_name" in sig.parameters


class TestUpstreamTyposFixed:
    def test_no_undefined_building_patches(self):
        # Upstream Example 5 referenced a variable that was never defined.
        assert "building_patches" not in coder_prompt.CODE
        assert "building_patches" not in coder_prompt.EXAMPLES_CODER

    def test_exists_has_a_spec_body(self):
        # It was listed in the method summary with no body to copy from.
        assert "def exists(" in coder_prompt.CODE

    def test_mask_attribute_is_documented(self):
        # grasp_detection reads object_patch.mask, so the LLM needs to know it.
        attrs = coder_prompt.CODE[: coder_prompt.CODE.index("Methods")]
        assert "mask" in attrs
