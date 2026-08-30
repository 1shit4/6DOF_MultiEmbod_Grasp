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


class TestVerticalConventionIsConsistent:
    """The two prompts disagreed about which way is up, and one of them was wrong.

    `find()` builds patches with `lower = height - ymax`, a bottom-origin axis,
    so a LARGER `vertical_center` is HIGHER in the picture. The Coder prompt said
    so; the Planner prompt called it "image rows", which implies the opposite.
    The Planner writes the plan the Coder implements, so a disagreement here
    silently grasps the object at the wrong end of the table with no error.
    """

    def test_the_code_really_is_bottom_origin(self):
        """Pin the convention to the implementation, not to either prompt."""
        import inspect

        import image_patch

        src = inspect.getsource(image_patch.ImagePatch.find)
        assert "self.height - min(self.height" in src, (
            "find() no longer flips the y axis; the prompt guidance below "
            "and both prompts must be revisited"
        )

    def test_planner_no_longer_calls_it_image_rows(self):
        assert "(image rows)" not in planner_prompt.PLAN

    def test_planner_states_larger_is_higher(self):
        assert re.search(
            r"LARGER value is HIGHER", planner_prompt.PLAN
        ), "the Planner prompt must state the bottom-origin convention"

    def test_coder_states_larger_is_higher(self):
        assert re.search(
            r"higher values are closer to the top", coder_prompt.CODE, re.I
        )

    def test_coder_lower_border_is_not_described_as_bottom_ward(self):
        """`lower` is measured from the bottom, so a bigger value is nearer the
        TOP. The attribute block used to contradict its own vertical_center
        line four lines further down."""
        block = coder_prompt.CODE.split("Methods")[0]
        lower = block.split("lower : int")[1].split("right : int")[0]
        assert "closer to the bottom" not in lower.lower()

    def test_both_prompts_agree_on_which_index_is_highest(self):
        """One convention, and both prompts must point at the same index.

        Checks the pairing rather than the wording: in each prompt, the first
        index token after the word "highest" has to be [-1]. That is what the
        bottom-origin axis makes true, and what the old "image rows" phrasing
        got backwards.
        """
        for name, text in (
            ("planner", planner_prompt.PLAN), ("coder", coder_prompt.CODE)
        ):
            after = text.lower().find("highest object")
            assert after != -1, f"{name} prompt never says which object is highest"
            token = re.search(r"\[-1\]|\[0\]", text[after:])
            assert token is not None, f"{name} prompt names no index for 'highest'"
            assert token.group(0) == "[-1]", (
                f"{name} prompt ties 'highest' to {token.group(0)}; with "
                "vertical_center measured from the bottom it must be [-1]"
            )


class TestFailureTaxonomy:
    """Geometry, wrong part and wrong object need different people to fix them.

    A single INVALID verdict conflates three failures with three different
    owners. Geometry and wrong-part the inner loop can fix by re-planning;
    wrong-object it cannot, because the instance id is chosen by the outer task
    planner before the inner loop begins. Telling the Planner to "diversify" on
    a wrong-object failure invites it to grasp something it was never asked for.
    """

    def test_observer_emits_a_failure_kind(self):
        assert '"failure"' in observer_prompt.OBSERVER
        for kind in ("geometry", "wrong_part", "wrong_object"):
            assert kind in observer_prompt.OBSERVER

    def test_observer_says_wrong_object_is_not_fixable_here(self):
        assert re.search(
            r"wrong_object.{0,400}(cannot fix|Nobody in this\s+loop)",
            observer_prompt.OBSERVER, re.S | re.I,
        )

    def test_planner_is_told_not_to_switch_objects(self):
        assert re.search(
            r"wrong_object.{0,300}(Do not switch objects|cannot fix)",
            planner_prompt.PLAN, re.S | re.I,
        )

    def test_every_checklist_field_can_decide_the_verdict(self):
        """Two of the five used to be computed and then ignored by the rule.

        `target_match` and `collision_risk` were absent from the decision rule,
        so the Observer filled them in and they changed nothing.
        """
        rule = observer_prompt.OBSERVER.split("Decision Rule")[1][:600]
        for field in (
            "target_match", "semantic_alignment", "fragile_overlap",
            "collision_risk", "approach_feasibility",
        ):
            assert field in rule, f"{field} cannot affect the verdict"


class TestObserverIsGivenTheMeasurements:
    def test_it_is_told_about_the_selection_funnel(self):
        assert "selection" in observer_prompt.OBSERVER
        assert "clear_of_the_rest_of_the_scene" in observer_prompt.OBSERVER

    def test_it_is_told_to_trust_the_numbers_over_the_picture(self):
        """The projection cannot resolve depth ordering; the 3D test already did."""
        assert re.search(
            r"[Tt]rust (them|the numbers).{0,120}over the picture",
            observer_prompt.OBSERVER, re.S,
        )

    def test_it_is_given_the_gripper_morphology(self):
        assert "gripper_morphology" in observer_prompt.OBSERVER
        assert "n_fingers" in observer_prompt.OBSERVER

    def test_the_summary_actually_carries_morphology(self):
        """The prompt must not describe a field the code never sends."""
        import inspect

        from agents import graspmas

        src = inspect.getsource(graspmas.GraspMAS._build_result)
        assert '"gripper_morphology"' in src
        assert '"selection"' in src


class TestGraspPurposeIsAPrinciple:
    """The purpose branch must state a principle, not a list of objects.

    An earlier version of this prompt was tuned against a live run and said
    things like "a mug's handle is a weaker hold" and "do NOT call find_part".
    Both were wrong, and wrong in the way that matters: the evidence was a
    *synthetic* scene whose "mug" is a bare cylinder and whose "banana" is a
    box, so VLPart was right to find no parts and there was nothing to fix.
    Baking object-specific verdicts into the prompt trades the model's own
    judgement for a handful of cases we happened to observe, and breaks on the
    unseen object that is the whole point of using a model here.
    """

    @staticmethod
    def _branch(marker):
        return planner_prompt.PLAN.split(marker)[1][:700]

    def test_declutter_states_a_principle_not_a_parts_list(self):
        branch = self._branch("in order to move it out of the way")
        assert re.search(r"most secure and least\s+likely\s+to damage", branch, re.I)

    def test_declutter_does_not_forbid_find_part(self):
        """Part choice is central to grasping, and the Observer's `wrong_part`
        correction has nowhere to go if the tool is banned."""
        branch = self._branch("in order to move it out of the way")
        assert not re.search(r"do NOT call `find_part`", branch, re.I)

    def test_no_object_specific_verdicts_are_baked_in(self):
        branch = self._branch("in order to move it out of the way")
        for tuned in ("mug's handle", "jug's spout", "pan's grip", "weaker hold"):
            assert tuned not in branch, f"prompt is tuned to a specific object: {tuned}"

    def test_handover_generalises_beyond_the_example(self):
        """The knife may illustrate the rule; it may not BE the rule."""
        branch = self._branch("so it can be handed to the person")
        assert re.search(r"should NOT receive", branch)
        assert re.search(r"no example here covers", branch, re.I)

    def test_the_fallback_is_not_a_named_part(self):
        """"default to the most secure hold (the handle)" was a hidden verdict."""
        assert "(the handle)" not in planner_prompt.PLAN

    def test_wrong_part_still_routes_to_find_part(self):
        """The Observer's correction path must remain reachable."""
        assert re.search(
            r"wrong_part.{0,200}find_part", planner_prompt.PLAN, re.S
        )


class TestReselectionIsReachable:
    """The Coder cannot use an affordance it has never been told about.

    Its API doc said grasp_detection returns "the best" grasp and nothing else,
    so writing code that picks a different one was not a failure of imagination
    — it was correct behaviour against the only tool it had been shown. The
    whole reason the Coder is a language model rather than a fixed script is
    that it can compose a filter on the spot; that needs something to filter.
    """

    def test_the_coder_is_told_candidates_exist(self):
        assert "candidates" in coder_prompt.CODE
        assert "select_grasp" in coder_prompt.CODE

    def test_the_coder_has_a_worked_reselection_example(self):
        assert re.search(
            r"select_grasp\(grasp_pose, alternatives\)", coder_prompt.EXAMPLES_CODER
        )

    def test_the_coder_is_warned_that_recomputing_repeats_itself(self):
        assert re.search(
            r"re-running grasp_detection.{0,120}same pose",
            coder_prompt.CODE, re.S | re.I,
        )

    def test_the_planner_knows_a_rejection_has_a_remedy(self):
        assert "select_grasp" in planner_prompt.PLAN
        assert re.search(
            r"geometry.{0,400}change the GRASP", planner_prompt.PLAN, re.S
        )

    def test_the_planner_is_told_not_to_just_repeat_itself(self):
        assert re.search(
            r"same computation on the same\s+mask", planner_prompt.PLAN, re.S | re.I
        )

    def test_select_grasp_exists_on_the_class(self):
        """The prompt must not advertise a method the code does not have."""
        from image_patch import ImagePatch

        assert hasattr(ImagePatch, "select_grasp")
