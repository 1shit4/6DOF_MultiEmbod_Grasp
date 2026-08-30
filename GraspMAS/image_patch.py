from __future__ import annotations
from matplotlib import pyplot as plt
import base64
import logging
import os
import re
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
from typing import List, Optional, Tuple
from scipy import spatial
from torchvision import transforms
from torchvision.ops import box_convert
from transformers import (DPTImageProcessor, DPTForDepthEstimation,
                          AutoModelForMaskGeneration, AutoProcessor, pipeline)
from vlpart.vlpart import build_vlpart
import detectron2.data.transforms as T

from agents.llm import get_shared_llm
from collision import (
    filter_grasps_by_scene_collision,
    gripper_geometry,
    load_gripper_points,
    scene_cloud_excluding,
)
from graspgen import GraspGenClient, GraspGenUnavailable
from perception3d import (
    MAX_CLOUD_POINTS,
    MIN_OBJECT_POINTS,
    Grasp6D,
    SceneContext,
    downsample_cloud,
    filter_grasps_by_mask,
    filter_grasps_by_visibility,
    grasp_to_rect2d,
    unproject,
)

logger = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## GroundingDINO
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
THRESHOLD = 0.4
detector_id = 'IDEA-Research/grounding-dino-tiny'
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

## SAM model
segmenter_id = "facebook/sam-vit-base"
segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
segmentor_processor = AutoProcessor.from_pretrained(segmenter_id)

## VLpart model (part-level grounding: "the handle of the knife")
# Anchored to BASE_PATH: a bare relative path only resolved when the process
# happened to be started from the GraspMAS directory.
vlpart_checkpoint = os.environ.get(
    "VLPART_CHECKPOINT", os.path.join(BASE_PATH, "weights", "swinbase_part_0a0000.pth")
)
if not os.path.exists(vlpart_checkpoint):
    raise FileNotFoundError(
        f"VLPart checkpoint not found at {vlpart_checkpoint}. "
        f"Run `bash download.sh` from the GraspMAS directory, or set "
        f"$VLPART_CHECKPOINT."
    )
vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
vlpart.to(device=device)
THRESHOLD_VLPART = 0.3

## MiDaS — relative inverse depth. Only a fallback ordering cue now: it cannot
## be unprojected, so metric depth for 6-DoF comes from SCENE (see perception3d).
depth_id = "Intel/dpt-hybrid-midas"
depth_processor = DPTImageProcessor.from_pretrained(depth_id)
depth_model = DPTForDepthEstimation.from_pretrained(depth_id, low_cpu_mem_usage=True)

# NOTE ON REMOVED MODELS
# OWLv2 and BLIP2-flan-t5-XL used to be loaded here at import time. OWLv2 had
# zero call sites anywhere in the repo, and BLIP2 (15.8 GB download, ~8 GB
# resident, with a hardcoded .to(device="cuda")) was reachable only from
# `best_image_match`, which no agent, prompt or example ever calls. Dropping
# both is what makes this module fit on a CPU-only 15 GB machine.
# The RAGT-3-3 planar grasp model is gone too — GraspGen-X replaces it.

# ---------------------------------------------------------------------------
# Per-query state
# ---------------------------------------------------------------------------

# Metric depth + intrinsics for the image under analysis. The LLM writes
# `execute_command(image)` which receives only the RGB array, so calibration
# has to reach `grasp_detection` through module state. `GraspMAS.query` sets it.
SCENE: SceneContext = SceneContext()

# Where grasp visualizations and per-call artifacts go; set per query.
RECORDER = None
SAVE_FOLDER = "imgs/"

GRIPPER_NAME = "franka_panda"
NUM_GRASPS = 200
PLANNER = "graspmoe"

_GRASPGEN_CLIENT: Optional[GraspGenClient] = None
# Keyed by (mask bytes hash, gripper, planner, num_grasps) — a re-planned round
# often reuses the same mask, and on CPU a cache hit saves ~10 s per call.
#
# Holds the whole surviving *candidate set*, not the single pose picked from it.
# It used to hold the winner, which made a re-plan after a rejection return a
# bit-identical grasp: same mask, same key, same answer, and no way to ask for a
# different one. See `_SELECTION_REGISTRY` below.
_GRASP_CACHE: dict = {}

# Masks that produced a grasp, so the overlay can tint the region the grasp was
# actually computed for. The Observer's prompt describes that magenta region and
# it was never drawn, because the mask lives in `grasp_detection` and the result
# dict carries no way back to it.
#
# Keyed by a monotonic int stored on the result rather than by `id()` (which is
# reused after a GC) or by "the last mask seen" (which is wrong whenever the
# generated code computes a part grasp and an object grasp in one round). The
# key is a plain int, so it serialises with the rest of the result harmlessly.
_MASK_REGISTRY: dict = {}
_MASK_SEQ = 0
_MASK_REGISTRY_MAX = 64


def _register_mask(mask: np.ndarray) -> int:
    global _MASK_SEQ
    _MASK_SEQ += 1
    if len(_MASK_REGISTRY) >= _MASK_REGISTRY_MAX:
        for stale in sorted(_MASK_REGISTRY)[: _MASK_REGISTRY_MAX // 2]:
            _MASK_REGISTRY.pop(stale, None)
    _MASK_REGISTRY[_MASK_SEQ] = mask
    return _MASK_SEQ


# The surviving candidate set behind each returned grasp, so a rejected grasp
# can be replaced by a *different* one without asking the server again.
#
# GraspGen-X generates ~416 poses per call and 16-66 of them survive every
# filter, with the runners-up typically within 0.02-0.05 of the winner. All but
# one were discarded, so the only answer the loop had to "this grasp is wrong"
# was to re-run the identical pipeline and get the identical pose back.
#
# Same monotonic-int handle as the mask registry, for the same reason: the
# result dict has to be JSON-serialisable, so it carries a key rather than the
# arrays themselves.
_SELECTION_REGISTRY: dict = {}
_SELECTION_SEQ = 0
_SELECTION_REGISTRY_MAX = 16

#: How many alternatives to hand back. Enough to re-choose from, few enough
#: that the result stays a reasonable size when it is written to disk.
MAX_RETURNED_CANDIDATES = 20


def _register_selection(selection: dict) -> int:
    global _SELECTION_SEQ
    _SELECTION_SEQ += 1
    if len(_SELECTION_REGISTRY) >= _SELECTION_REGISTRY_MAX:
        for stale in sorted(_SELECTION_REGISTRY)[: _SELECTION_REGISTRY_MAX // 2]:
            _SELECTION_REGISTRY.pop(stale, None)
    _SELECTION_REGISTRY[_SELECTION_SEQ] = selection
    return _SELECTION_SEQ


def _build_grasp_result(sel: dict, chosen: int) -> dict:
    """Assemble the returned grasp dict for one candidate of a selection.

    Split out of `grasp_detection` so that choosing again costs nothing but a
    dictionary rebuild — no depth, no unprojection, no server round trip.
    """
    grasps, scores = sel["grasps"], sel["scores"]
    width, fingertip_depth = sel["width"], sel["fingertip_depth"]
    grasp = Grasp6D(
        pose=grasps[chosen],
        score=float(scores[chosen]),
        gripper=sel["gripper"],
        width=width,
        rect_2d=grasp_to_rect2d(
            grasps[chosen], sel["K"], width, float(scores[chosen]),
            fingertip_depth=fingertip_depth,
        ),
    )
    result = grasp.as_dict()
    result["mask_key"] = sel["mask_key"]
    result["selection_key"] = sel["selection_key"]
    result["filters"] = dict(sel["filters"])
    result["chosen_index"] = int(chosen)

    # The alternatives, best first: enough for generated code to filter on
    # (score, where the hand sits, which way it comes in) without carrying a
    # 4x4 matrix for each. The pose of whichever one is picked is recovered
    # from the registry by `select_grasp`.
    keep = sel["keep"]
    order = keep[np.argsort(scores[keep])[::-1]][:MAX_RETURNED_CANDIDATES]
    result["candidates"] = [
        {
            "index": int(i),
            "score": round(float(scores[i]), 4),
            "position": [round(float(v), 4) for v in grasps[i][:3, 3]],
            "approach": [round(float(v), 4) for v in grasps[i][:3, 2]],
            "closing": [round(float(v), 4) for v in grasps[i][:3, 0]],
        }
        for i in order
    ]
    return result


#: How many of the best rejected candidates to attribute. Enough to see which
#: object is responsible, few enough that the sweep stays cheap — each one costs
#: a collision test per instance in the scene.
MAX_ATTRIBUTED_CANDIDATES = 12


def _grasped_instances(mask: np.ndarray) -> set:
    """Registry instances the grasp mask actually covers.

    Identified by overlap rather than by the patch's name, because `find()`
    names a patch after a *label* while `find_by_id` names it after an id, and a
    part mask from `find_part` is named after neither. Overlap works for all
    three, including the part case: a knife handle's mask sits inside the
    knife's, so the knife is correctly recognised as the object being grasped.
    """
    instances = getattr(REGISTRY, "instances", None) if REGISTRY is not None else None
    if not instances:
        return set()
    got = set()
    for oid, inst in instances.items():
        if getattr(inst, "mask", None) is None:
            continue
        m = inst.mask
        if m.shape != mask.shape:
            continue
        area = int(m.sum())
        if area and int(np.logical_and(m, mask).sum()) > 0.5 * area:
            got.add(oid)
    return got


def _attribute_collisions(rejected_poses, gripper_name: str, exclude=()) -> list:
    """Which objects the rejected grasp candidates ran into, worst first.

    The scene cloud the filter tests against is anonymous — depth pixels with
    the target masked out — so it can say a candidate collided and never with
    what. Recovering that turns "no grasp found on the mug" into "the box is
    what stands between the hand and the mug", which is the difference between
    a planner that can act and one that can only give up.

    Needs the decluttering registry, so it is a no-op for one-shot queries.
    """
    if REGISTRY is None or not len(rejected_poses):
        return []
    tally: dict = {}
    for pose in rejected_poses[:MAX_ATTRIBUTED_CANDIDATES]:
        try:
            hits = REGISTRY.colliding_instances(
                pose, gripper_name=gripper_name, exclude=tuple(exclude)
            )
        except Exception as exc:  # a diagnostic must never break a working grasp
            logger.debug("could not attribute a collision: %s", exc)
            return []
        for oid, label, _n in hits:
            entry = tally.setdefault(oid, {"object_id": oid, "label": label,
                                           "grasps_fouled": 0})
            entry["grasps_fouled"] += 1
    return sorted(tally.values(), key=lambda e: -e["grasps_fouled"])


def grasp_mask(grasp: Optional[dict]) -> Optional[np.ndarray]:
    """The mask a grasp was computed from, if it is still held.

    Returns None rather than raising when the entry has been evicted — the
    overlay degrades to an untinted picture, which is what it always drew.
    """
    if not isinstance(grasp, dict):
        return None
    return _MASK_REGISTRY.get(grasp.get("mask_key"))


# The decluttering loop's instance registry, when one is running. It gives the
# generated code `find_by_id` (unambiguous with duplicate objects) and gives
# `grasp_detection` the rest of the scene to collision-check against.
REGISTRY = None


def set_registry(registry) -> None:
    """Attach or detach the scene registry for the current iteration.

    Separate from `configure_scene` because its lifetime is different: the
    registry spans the whole run and carries ids across iterations, while the
    scene context is rebuilt per capture.
    """
    global REGISTRY
    REGISTRY = registry
    _GRASP_CACHE.clear()


def configure_scene(
    scene: Optional[SceneContext] = None,
    recorder=None,
    save_folder: Optional[str] = None,
    gripper_name: Optional[str] = None,
    num_grasps: Optional[int] = None,
    planner: Optional[str] = None,
) -> None:
    """Publish per-query context for the generated code to pick up."""
    global SCENE, RECORDER, SAVE_FOLDER, GRIPPER_NAME, NUM_GRASPS, PLANNER
    if scene is not None:
        SCENE = scene
        _GRASP_CACHE.clear()
    if recorder is not None:
        RECORDER = recorder
    if save_folder is not None:
        SAVE_FOLDER = save_folder
    if gripper_name is not None:
        GRIPPER_NAME = gripper_name
        _GRASP_CACHE.clear()
    if num_grasps is not None:
        NUM_GRASPS = int(num_grasps)
    if planner is not None:
        PLANNER = planner


def _scene_cloud_excluding(mask) -> Optional[np.ndarray]:
    """The scene minus the object being grasped, for collision checking.

    Returns None when there is no depth to work from, in which case the caller
    skips the collision filter rather than rejecting everything.
    """
    try:
        depth = SCENE.depth
        if depth is None:
            return None
        K = SCENE.get_intrinsics(None) if SCENE.intrinsics is None else SCENE.intrinsics
        if K is None:
            return None
        return scene_cloud_excluding(depth, K, exclude_mask=mask)
    except Exception as exc:  # never let a filter break a working grasp path
        logger.debug("could not build a scene cloud for collision checking: %s", exc)
        return None


def get_graspgen_client() -> GraspGenClient:
    """Lazily connect to the GraspGen-X server (separate conda env, own process)."""
    global _GRASPGEN_CLIENT
    if _GRASPGEN_CLIENT is None:
        _GRASPGEN_CLIENT = GraspGenClient()
    return _GRASPGEN_CLIENT


def get_llm():
    return get_shared_llm(recorder=RECORDER)

def convert_bbox(image_source: torch.Tensor, boxes: torch.Tensor) -> np.ndarray:
    _,h, w = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy

def img_unormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
    var = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
    img = img*var + mean
    return torch.tensor(img*255, dtype=torch.uint8)

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

# CLIP backs `best_image_match`. Loaded lazily so the ~240 MB RN50 download only
# happens if that tool is actually used.
_clip_module = None
_CLIP_CACHE = None


def _get_clip():
    global _clip_module, _CLIP_CACHE
    if _CLIP_CACHE is None:
        import clip as _clip
        _clip_module = _clip
        logger.info("Loading CLIP RN50 for best_image_match")
        _CLIP_CACHE = _clip.load("RN50", device=device)
    return _CLIP_CACHE


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median metric depth (metres) of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    overlaps_with(left: int, lower: int, right: int, upper: int)->bool
        Returns True if a crop with the given coordinates overlaps with this one, else False.
    find_part(object_name: str, part_name: str)->ImagePatch
        Returns a ImagePatch object of the part of the object.
    grasp_detection(object_patch: ImagePatch)->dict
        Returns the best 6-DoF grasp pose for the object/part in object_patch.
    """

    def __init__(self, image: Image.Image | torch.Tensor | np.ndarray, left: int | None = None, lower: int | None = None,
                 right: int | None = None, upper: int | None = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None, mask = None, name=""):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255


        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
            if mask == None:
                self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            else:
                self.mask = mask
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower
            # if mask.any() == None:
            #     self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            # else:
            #     self.mask = torch.from_numpy(mask[image.shape[1]-upper:image.shape[1]-lower, left:right])

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        # self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch
        self.mask = mask

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        # if self.height<700 or self.width<700:
        #     scale = max(700/self.width,700/self.height)
        #     self.height = int(self.height*scale)
        #     self.width = int(self.width*scale)
        #     self.cropped_image = torchvision.transforms.Resize((self.height,self.width))(self.cropped_image)
        #     self.mask = torchvision.transforms.Resize((self.height,self.width))(self.mask.unsqueeze(0)).squeeze()

        # draw bounding box and fill color
        # self.PIL_img = img_unormalize(self.cropped_image)
        # transform this image to PIL image
        self.PIL_img = torchvision.transforms.ToPILImage()(self.cropped_image)
        self.original_img = torchvision.transforms.ToPILImage()(image)
        self.original_width = image.shape[2]
        self.original_height = image.shape[1]

        # `name` was accepted by the constructor but never stored, so callers
        # had no way to tell which object a patch referred to. grasp_detection
        # uses it to label the artifacts it writes.
        self.name = name or ""
        # Set by find_part: False when VLPart had no such part and the whole
        # object was substituted. None for patches that are not parts.
        self.part_found = None

        # Upstream wrote straight into BASE_PATH/imgs/ and crashed if that
        # directory did not exist, and used the raw object name in the filename
        # (so a name like "knife blade/handle" produced an invalid path).
        img_dir = os.path.join(BASE_PATH, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:60]
        filename = f"query_image_{safe}.png" if safe else "query_image.png"
        self.query_image_path = os.path.join(img_dir, filename)
        self.PIL_img.save(self.query_image_path, "PNG")


    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        labels = [label if label.endswith(".") else label+"." for label in [object_name]]
        image = self.PIL_img
        detection_results = object_detector(image, candidate_labels=labels, threshold=THRESHOLD)
        boxes = [[result["box"]['xmin'], 
                  result["box"]['ymin'], 
                  result["box"]['xmax'], 
                  result["box"]['ymax']] 
                for result in detection_results]
        if len(boxes) == 0:
            return [None]
        inputs = segmentor_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
        outputs = segmentator(**inputs)
        masks = segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        masks = refine_masks(masks, polygon_refinement=True)
        obj_list = []
        for mask, box in zip(masks, boxes):
            # padding 10% of the width and height of the image patch. We find it better to do it this way for VLpart.
            scale = 0.1
            max_d = max(int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
            left = max(0, int(box[0]) - scale*max_d)
            right = min(self.width, int(box[2]) + scale*max_d)
            lower = self.height - min(self.height, int(box[3]) + scale*max_d)
            upper = self.height - max(0, int(box[1]) - scale*max_d)
            obj = self.crop(left, lower, right, upper, mask=mask, object_name=object_name)
            obj_list.append(obj)
        return obj_list

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        prompt = f"Question: Do you see a {object_name} in the image? Answer (Y/N):"
        answer = get_llm().sync_chat_with_image(
            "Answer with a single letter: Y or N.",
            prompt,
            self._image_b64(),
            agent="exists",
            max_tokens=5,
        )
        # The original compared `== 'Y'`, which fails on "Y." or "Yes" — both
        # of which other models emit routinely.
        return answer.strip().upper().startswith("Y")

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        
        prompt = f"Is the {object_name} {attribute}? Answer yes or no?"
        answer = get_llm().sync_chat_with_image(
            "Answer with a single word: yes or no.",
            prompt,
            self._image_b64(),
            agent="verify_property",
            max_tokens=5,
        )
        return "yes" in answer.lower()

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        return get_llm().sync_chat_with_image(
            "Answer the question about the image concisely.",
            question,
            self._image_b64(),
            agent="simple_query",
            max_tokens=300,
        )

    def _image_b64(self) -> str:
        """This patch as a base64 PNG, for the vision tool calls.

        Encodes the patch itself rather than re-reading the whole query image
        from disk (which is what the original did), so a question about a
        cropped patch is actually asked about that crop.
        """
        arr = np.array(self.PIL_img)
        ok, buf = cv2.imencode(".png", arr[:, :, ::-1])
        if not ok:
            raise RuntimeError("Failed to encode patch as PNG")
        return base64.b64encode(buf).decode("utf-8")

    def best_image_match(self, list_patches: list[ImagePatch], content: list[str], return_index: bool = False) -> \
            ImagePatch | int | None:
        """Returns the patch most likely to contain the content.
        Parameters
        ----------
        list_patches : List[ImagePatch]
        content : List[str]
            the object of interest<
        return_index : bool
            if True, returns the index of the patch most likely to contain the object

        Returns
        -------
        int
            Patch most likely to contain the object
        """

        if len(list_patches) == 0:
            return None
        if isinstance(content, str):
            content = [content]

        # Originally: BLIP2 captions scored against paid OpenAI text embeddings.
        # CLIP does image-text matching directly, is already a dependency (VLPart
        # needs it), runs on CPU, and costs nothing.
        model, preprocess = _get_clip()
        images = torch.stack(
            [preprocess(patch.PIL_img) for patch in list_patches]
        ).to(device)
        tokens = _clip_module.tokenize(list(content)).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores = (image_features.float() @ text_features.float().T).mean(dim=1)
        best = int(scores.argmax().item())

        if return_index:
            return best
        return list_patches[best]

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def crop(self, left: int, lower: int, right: int, upper: int, mask, object_name) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)
        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self, mask = mask, name=object_name)


    def llm_query(self, question, context=None, long_answer=True, queues=None):
        """Answers a question using external world knowledge from the LLM.

        Parameters
        ----------
        question: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        system = (
            "Answer using general world knowledge."
            if long_answer
            else "Answer concisely, in a few words, using general world knowledge."
        )
        prompt = question if not context else f"{context}\n\n{question}"
        return get_llm().sync_chat(
            system, prompt, agent="llm_query", max_tokens=300 if long_answer else 60
        )

    def bool_to_yesno(self, bool_answer: bool) -> str:
        """Returns a yes/no answer to a question based on the boolean value of bool_answer.
        Parameters
        ----------
        bool_answer : bool
            a boolean value

        Returns
        -------
        str
            a yes/no answer to a question based on the boolean value of bool_answer
        """
        return "yes" if bool_answer else "no"

    def compute_depth(self):
        """Returns the median distance from the camera to this patch, in METRES.

        Smaller is closer. Note this inverts the old behaviour: this method used
        to return MiDaS relative *inverse* depth cast to int, where larger meant
        closer and the value could not be unprojected into 3D. It is now real
        metric depth taken from the same source `grasp_detection` uses, so
        "nearest object" and "furthest object" mean what they say and are
        consistent with the 6-DoF geometry.

        Returns
        -------
        float
            median depth of the patch in metres
        """
        full_image = np.array(
            self.PIL_img if self.parent_img_patch is None
            else self.parent_img_patch.PIL_img
        )
        try:
            depth_map = SCENE.get_depth(full_image)
        except Exception as exc:
            logger.warning("compute_depth: metric depth unavailable (%s); "
                           "falling back to relative MiDaS", exc)
            return self._relative_depth()

        # This patch's window into the full image. ImagePatch stores `upper`/
        # `lower` in a bottom-left origin, so flip to array row indices.
        h = full_image.shape[0]
        top = max(0, h - int(self.upper))
        bottom = min(h, h - int(self.lower))
        left = max(0, int(self.left))
        right = min(full_image.shape[1], int(self.right))
        if bottom <= top or right <= left:
            region = depth_map
            region_mask = None
        else:
            region = depth_map[top:bottom, left:right]
            region_mask = None
            if self.mask is not None:
                m = np.asarray(self.mask)
                if m.shape[:2] == depth_map.shape[:2]:
                    region_mask = m.astype(bool)[top:bottom, left:right]

        valid = np.isfinite(region) & (region > 0)
        if region_mask is not None and region_mask.any():
            valid &= region_mask
        if not valid.any():
            logger.warning("compute_depth: no valid depth in patch; using MiDaS")
            return self._relative_depth()
        return float(np.median(region[valid]))

    def _relative_depth(self) -> float:
        """MiDaS relative inverse depth — ordering only, never unprojected."""
        image = self.PIL_img
        inputs = depth_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            predicted_depth = depth_model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        # MiDaS is inverse depth (larger = closer). Negate so the sign convention
        # matches metric depth (smaller = closer) and sorting stays consistent
        # whichever branch produced the value.
        return -float(np.median(prediction.squeeze().cpu().numpy()))

    #: How many alternative names to try when the requested one scores nothing.
    #: Two, not "until something matches". A wider search is slower, spends
    #: requests, and — the real cost — makes it likelier that a term matching
    #: *some* region gets accepted as the region that was asked for. The
    #: Observer sees the substitution and judges whether it was the right one.
    MAX_PART_SYNONYMS = 2

    def _vlpart_query(self, image_np, inputs, text_prompt):
        """Run VLPart for one phrasing. Returns (boxes above threshold, best score).

        The `"{object} {part}"` two-word shape is load-bearing and measured:
        on the mustard bottle in `real_world/00`, `bottle cap` scores 0.590,
        a bare `cap` scores 0.003 and `the cap of the bottle` 0.062. Do not
        "improve" this into a sentence.
        """
        with torch.no_grad():
            predictions = vlpart.inference([inputs], text_prompt=text_prompt)[0]

        filter_boxes, best = [], 0.0
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            if scores is not None and len(scores):
                best = float(scores.max())
                for obj_ind in range(len(scores)):
                    if float(scores[obj_ind]) < THRESHOLD_VLPART:
                        continue
                    filter_boxes.append(boxes[obj_ind])
        return filter_boxes, best

    def _part_synonyms(self, object_name: str, part_name: str) -> list:
        """Other names for the same physical region, from the model. Best effort.

        Asked for rather than tabulated: a hardcoded synonym list only covers
        the objects whoever wrote it thought of, and the point of having a
        language model in the loop is that it knows the ones we did not.

        The wording insists on the SAME region, because a plausible synonym is
        not automatically the same piece of the object — "grip" and "handle"
        coincide on a mug and not on a pair of scissors. That risk is why the
        result is reported as a substitution rather than silently accepted.
        """
        try:
            reply = self.llm_query(
                f"A vision system could not locate the '{part_name}' of a "
                f"{object_name}. Give up to {self.MAX_PART_SYNONYMS} alternative "
                f"single-word names for the SAME physical region of a "
                f"{object_name}. Reply with only the words, comma-separated, no "
                f"explanation. If there is no other name for it, reply NONE.",
                long_answer=False,
            )
        except Exception as exc:  # no key, offline, provider down — not fatal
            logger.debug("find_part: could not ask for alternative names: %s", exc)
            return []

        out, seen = [], {part_name.strip().lower()}
        for token in re.split(r"[,\n;]+", str(reply or "")):
            word = token.strip().strip(".\"'").lower()
            if not word or word == "none" or word in seen:
                continue
            if len(word.split()) > 2:  # a sentence, not a name
                continue
            seen.add(word)
            out.append(word)
            if len(out) >= self.MAX_PART_SYNONYMS:
                break
        return out

    def find_part(self, object_name: str, part_name: str) -> ImagePatch:
        """Returns a ImagePatch object of the part of the object

        A failed lookup is usually a naming problem rather than an absent part:
        VLPart is open-vocabulary, and on the same bottle `cap` scores 0.590
        while `lid` scores 0.058 — a tenfold gap for one physical feature. So
        the requested term is tried first and, only if it finds nothing, up to
        `MAX_PART_SYNONYMS` alternatives are asked for and tried.

        Whatever happens is reported on the returned patch rather than hidden:
        `part_found`, `part_term` (what actually matched), `part_substituted`
        and `part_terms_tried`. A substitution is a *guess about wording* and
        may well be a guess about the wrong region, so the Observer is told and
        can reject it.

        Parameters
        ----------
        object_name:
            name of the object
        part_name:
            name of the part of the object

        Returns
        -------
        ImagePatch
            a ImagePatch object of the part of the object (in size of original image)
        """
        image_np = np.array(self.PIL_img)
        preprocess = T.ResizeShortestEdge([800, 800], 1333)
        height, width = image_np.shape[:2]
        image = preprocess.get_transform(image_np).apply_image(image_np)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}

        tried, scores_seen = [], {}
        filter_boxes, matched = [], part_name
        for term in [part_name]:
            tried.append(term)
            filter_boxes, best = self._vlpart_query(
                image_np, inputs, f"{object_name} {term}"
            )
            scores_seen[term] = round(best, 3)
            if filter_boxes:
                matched = term
                break

        if not filter_boxes:
            for term in self._part_synonyms(object_name, part_name):
                tried.append(term)
                filter_boxes, best = self._vlpart_query(
                    image_np, inputs, f"{object_name} {term}"
                )
                scores_seen[term] = round(best, 3)
                if filter_boxes:
                    matched = term
                    logger.info(
                        "find_part: no '%s' on the %s, but '%s' matched (%.2f)",
                        part_name, object_name, term, best,
                    )
                    break

        if len(filter_boxes) == 0:
            # The fallback stays — a whole-object grasp beats no grasp — but it
            # is announced and flagged, so a caller can tell a part-level grasp
            # from an object-level one. Part-level grounding is the headline
            # claim; silently dropping it made the claim unverifiable.
            logger.warning(
                "find_part: VLPart found no '%s' on the %s above threshold %.2f "
                "(tried %s, best scores %s); falling back to the whole object",
                part_name, object_name, THRESHOLD_VLPART, tried, scores_seen,
            )
            fallback = self.crop(0, 0, self.width, self.height, mask=self.mask,
                                 object_name=f"{object_name}_{part_name}")
            fallback.part_found = False
            fallback.part_term = None
            fallback.part_substituted = False
            fallback.part_terms_tried = dict(scores_seen)
            return fallback

        inputs = segmentor_processor(images=image_np, input_boxes=[[filter_boxes]], return_tensors="pt").to(device)

        outputs = segmentator(**inputs)
        masks = segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement=True)[0] # numpy unit8
        # return to oringinal size
        mask_original = np.zeros_like(self.mask)
        mask_original[self.original_height-self.upper:self.original_height-self.lower, self.left:self.right] = masks
        left, lower, right, upper = map(int, filter_boxes[0])
        obj = self.crop(left, self.height-upper, right, self.height-lower, mask=mask_original, object_name=f"{object_name}_{part_name}")
        obj.part_found = True
        obj.part_term = matched
        obj.part_substituted = matched != part_name
        obj.part_terms_tried = dict(scores_seen)
        return obj

    def grasp_detection(self, object_patch, gripper_name: str = None, round_idx: int = 0):
        """Return the best 6-DoF grasp pose for the object/part in object_patch.

        Parameters
        ----------
        object_patch : ImagePatch
            The object or object part to grasp. Its `mask` selects which pixels
            are lifted into 3D, so a part-level patch from `find_part` yields a
            part-level grasp.
        gripper_name : str, optional
            Which robot gripper to generate grasps for (e.g. "franka_panda",
            "robotiq_2f_85", "unitree_g1"). Defaults to the configured gripper.

        Returns
        -------
        dict or None
            A 6-DoF grasp with keys:
              pose      4x4 homogeneous transform, camera frame, metres
              position  [x, y, z] gripper base position in metres
              approach  unit vector the gripper advances along (+Z of the pose)
              closing   unit vector the fingers close along (+X of the pose)
              score     grasp confidence in [0, 1]
              width     jaw aperture in metres
              gripper   the gripper name used
              rect_2d   [score, x, y, w, h, angle] projection into the image
            Returns None if no valid grasp could be produced.

        Pipeline: mask -> metric depth -> camera-frame point cloud ->
        GraspGen-X diffusion model -> mask-containment filter -> best by score.
        """
        if object_patch is None or isinstance(object_patch, list):
            return None

        mask = getattr(object_patch, "mask", None)
        if mask is None:
            logger.warning("grasp_detection: patch has no mask")
            return None

        image = np.array(self.PIL_img if self.parent_img_patch is None
                         else self.parent_img_patch.PIL_img)
        # Masks are stored at full-image resolution by find/find_part/crop.
        mask = np.asarray(mask)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = mask.astype(bool)
        if not mask.any():
            logger.warning("grasp_detection: empty mask")
            return None

        gripper = gripper_name or GRIPPER_NAME
        cache_key = (hash(mask.tobytes()), gripper, PLANNER, NUM_GRASPS)
        cached = _GRASP_CACHE.get(cache_key)
        if cached is not None:
            logger.info("grasp_detection: cache hit for %s", gripper)
            best = int(cached["keep"][int(np.argmax(cached["scores"][cached["keep"]]))])
            return _build_grasp_result(cached, best)

        try:
            depth = SCENE.get_depth(image)
            K = SCENE.get_intrinsics(image)
        except Exception as exc:
            logger.error("grasp_detection: could not resolve depth/intrinsics: %s", exc)
            return None

        cloud = unproject(depth, K, mask)
        raw_points = len(cloud)
        # Cap the cloud before it leaves this process: the server's outlier
        # filter is O(N^2) in the point count, and a close-up mask on a 720p
        # depth image routinely yields 40k+ points.
        cloud = downsample_cloud(cloud, MAX_CLOUD_POINTS)
        if raw_points != len(cloud):
            logger.info("Downsampled cloud %d -> %d points", raw_points, len(cloud))
        if len(cloud) < MIN_OBJECT_POINTS:
            logger.warning(
                "grasp_detection: only %d valid 3D points (need %d); "
                "the mask may cover a region with no depth return",
                len(cloud), MIN_OBJECT_POINTS,
            )
            return None

        if RECORDER is not None:
            RECORDER.save_mask(mask, getattr(object_patch, "name", "") or "object", round_idx)
            RECORDER.save_cloud(cloud, getattr(object_patch, "name", "") or "object", round_idx)

        try:
            client = get_graspgen_client()
            grasps, scores = client.infer(
                cloud,
                gripper_name=gripper,
                num_grasps=NUM_GRASPS,
                grasp_threshold=-1.0,
                topk_num_grasps=-1,
                planner=PLANNER,
            )
        except GraspGenUnavailable as exc:
            # A transient socket problem should not kill the agent loop; the
            # Observer will see "no grasp" and the Planner can retry.
            logger.error("grasp_detection: GraspGen-X unavailable: %s", exc)
            return None

        if len(grasps) == 0:
            logger.warning("grasp_detection: model returned no grasps")
            return None

        width, fingertip_depth = self._gripper_geometry(gripper)

        # Two filters, applied in order, each removing a different failure mode.
        #
        # 1. Visibility. GraspMoE sweeps candidate poses over every face of the
        #    fitted box, including the far side this single view never saw, so
        #    a large fraction of candidates approach from behind the object.
        n_all = len(grasps)
        visible = filter_grasps_by_visibility(grasps)
        if len(visible) == 0:
            logger.warning("grasp_detection: no grasp approaches from the "
                           "observed side; keeping all %d", n_all)
            visible = np.arange(n_all)

        # 2. Language. Geometry alone does not respect the referring expression:
        #    given a handle point cloud the model can still propose a pose that
        #    reaches back over the blade. Requiring the grasp centre to project
        #    inside the referred mask is what keeps language attached to the
        #    6-DoF output.
        in_mask = filter_grasps_by_mask(
            grasps[visible], scores[visible], mask, K, width,
            fingertip_depth=fingertip_depth,
        )
        keep = visible[in_mask] if len(in_mask) else visible
        fell_back_to_score = len(in_mask) == 0
        if fell_back_to_score:
            logger.warning(
                "grasp_detection: none of %d visible grasps landed inside the "
                "mask; ranking by score alone", len(visible)
            )

        # 3. Clutter. GraspGen-X is given one object's cloud and knows nothing
        #    about its neighbours, so on a crowded table it will happily propose
        #    a pose whose fingers pass straight through the mug alongside. On an
        #    isolated object this filter is a no-op; in clutter it is the
        #    difference between a plan and a collision.
        n_on_target = len(keep)
        every_candidate_collides = False
        collision_checked = False
        fouled_by: list = []
        scene_cloud = _scene_cloud_excluding(mask)
        if scene_cloud is not None and len(scene_cloud):
            collision_checked = True
            gripper_pts = load_gripper_points(gripper)
            clear = filter_grasps_by_scene_collision(
                grasps[keep], scene_cloud, gripper_pts, approach_len=0.08
            )
            # Which objects the *rejected* candidates ran into. This is what
            # replaces the old approach test, and it is a different kind of
            # answer: that test swept one synthesised top-down pose and asked
            # what fell inside it, so a hit said nothing about the grasps that
            # actually exist. This asks which objects stand between the robot
            # and the best poses the sampler really proposed, across every
            # direction it tried.
            rejected = np.setdiff1d(np.arange(len(keep)), clear, assume_unique=False)
            # The object being grasped is always inside the hand — that is what
            # a grasp is — so it must not be reported as standing in its own way.
            fouled_by = _attribute_collisions(
                grasps[keep[rejected]], gripper, exclude=_grasped_instances(mask)
            )

            if len(clear):
                keep = keep[clear]
            else:
                every_candidate_collides = True
                logger.warning(
                    "grasp_detection: all %d on-target grasps collide with the "
                    "rest of the scene; keeping them so the Observer can see why",
                    n_on_target,
                )

        logger.info(
            "grasp_detection: %d candidates -> %d visible -> %d on target -> %d clear",
            n_all, len(visible), n_on_target, len(keep),
        )

        best_local = int(np.argmax(scores[keep]))
        best_idx = int(keep[best_local])

        if RECORDER is not None:
            RECORDER.save_grasps(
                grasps, scores, getattr(object_patch, "name", "") or "object", round_idx,
                kept_indices=keep,
                meta={
                    "gripper": gripper,
                    "planner": PLANNER,
                    "num_points": int(len(cloud)),
                    "depth_source": SCENE.depth_source,
                    "n_in_mask": int(len(keep)),
                    "best_index": best_idx,
                },
            )

        # How the winner was arrived at. All of this was computed and then
        # logged to a file nobody reads: the Observer was asked to judge the
        # approach and the collision risk from a 2D projection while the exact
        # answers sat here. The comment on the fallback above literally says
        # "keeping them so the Observer can see why" — and it was never told.
        filters = {
            "n_candidates": int(n_all),
            "n_visible": int(len(visible)),
            "n_on_target": int(n_on_target),
            "n_clear_of_scene": int(len(keep)),
            "collision_checked": bool(collision_checked),
            # The two fallbacks. Each means "no candidate satisfied this test,
            # so the constraint was dropped rather than returning nothing" —
            # which makes the winner the least bad of a bad set, not a good one.
            "every_candidate_collides": bool(every_candidate_collides),
            "no_candidate_inside_mask": bool(fell_back_to_score),
            # False means find_part could not find the part and handed back the
            # whole object, so this grasp is not on the region asked for.
            "part_found": getattr(object_patch, "part_found", None),
            # True means the requested word found nothing and a different word
            # did. The region matched is a guess about naming and may not be the
            # region that was asked for, so the Observer is told rather than
            # left to assume the request was honoured.
            "part_substituted": getattr(object_patch, "part_substituted", False),
            "part_term": getattr(object_patch, "part_term", None),
            # Which objects the rejected candidates ran into. Empty when nothing
            # was rejected, or outside the decluttering loop where there is no
            # registry to attribute against.
            "fouled_by": fouled_by,
        }

        # Cache the whole surviving set, not the one pose picked from it. The
        # cache used to hold the single winner, which is why a re-plan after a
        # rejection returned a bit-identical grasp: same mask, same key, same
        # answer. Keeping the set means re-choosing is free and *different*.
        selection = {
            "grasps": grasps,
            "scores": scores,
            "keep": keep,
            "gripper": gripper,
            "width": width,
            "fingertip_depth": fingertip_depth,
            "K": K,
            "mask_key": _register_mask(mask),
            "filters": filters,
        }
        selection["selection_key"] = _register_selection(selection)
        _GRASP_CACHE[cache_key] = selection

        result = _build_grasp_result(selection, best_idx)
        logger.info(
            "grasp_detection: %s (+%d alternatives)",
            Grasp6D.from_dict(result).summary(), max(len(result["candidates"]) - 1, 0),
        )
        return result

    @staticmethod
    def select_grasp(grasp: dict, candidates) -> Optional[dict]:
        """Pick a different grasp from the ones `grasp_detection` already found.

        `grasp["candidates"]` holds the alternatives that survived every filter,
        best first. Filter that list however the situation calls for and pass
        what is left here; the highest-scoring survivor is returned as a full
        grasp, with no call to the grasp server.

        This is what makes an Observer rejection actionable. Without it the only
        reply to "that grasp is wrong" was to re-run the identical pipeline on
        the identical mask and get the identical pose back.

        Parameters
        ----------
        grasp : dict
            A result from `grasp_detection`.
        candidates : list[dict]
            A subset of `grasp["candidates"]` — or anything with an "index".

        Returns
        -------
        dict or None
            The best remaining grasp, or None if nothing was left. None means
            "every alternative was ruled out", which is worth reporting rather
            than working around.
        """
        sel = _SELECTION_REGISTRY.get((grasp or {}).get("selection_key"))
        if sel is None:
            logger.warning(
                "select_grasp: this grasp's candidate set is no longer held; "
                "call grasp_detection again"
            )
            return None

        wanted = []
        for c in candidates or []:
            idx = c.get("index") if isinstance(c, dict) else c
            if idx is None:
                continue
            wanted.append(int(idx))
        if not wanted:
            logger.warning("select_grasp: no candidates left after filtering")
            return None

        scores = sel["scores"]
        best = max(wanted, key=lambda i: float(scores[i]))
        return _build_grasp_result(sel, best)

    def find_by_id(self, object_id: str) -> "ImagePatch":
        """Return the registered instance `object_id` as a patch.

        Prefer this over `find` whenever the plan names an instance. `find`
        re-runs detection and returns a list whose order is not stable between
        iterations, so with two bottles in the scene "the bottle" can silently
        mean a different object each time. An id always means the same physical
        object for as long as it stays on the table.

        Examples
        --------
        >>> # Grasp the object the planner named
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     bottle = image_patch.find_by_id("obj_003")
        >>>     return image_patch.grasp_detection(bottle)
        """
        if REGISTRY is None:
            raise RuntimeError(
                "find_by_id needs a scene registry; it is only available inside "
                "the decluttering loop. Use find(object_name) otherwise."
            )
        inst = REGISTRY.get(object_id)
        ys, xs = np.nonzero(inst.mask)
        if len(xs) == 0:
            raise ValueError(f"{object_id} has an empty mask")

        # `crop` takes bottom-left-origin coordinates, so the row indices have to
        # be flipped against the full image height. Not `original_img.shape`:
        # that attribute is a PIL Image, which has no `.shape`, so every call
        # raised `'Image' object has no attribute 'shape'` and the Coder fell
        # back to `find()`. The registry's mask is full-frame, so it agrees with
        # `original_height` and covers a patch built without one.
        height = int(getattr(self, "original_height", 0)) or int(inst.mask.shape[0])
        patch = self.crop(
            int(xs.min()), int(height - ys.max()), int(xs.max()), int(height - ys.min()),
            inst.mask, inst.label,
        )
        patch.name = object_id
        return patch

    def place_detection(
        self,
        object_patch: "ImagePatch",
        grasp: dict,
        keep_clear: Optional[str] = None,
        gripper_name: str = None,
    ) -> Optional[dict]:
        """Where to put `object_patch` down after grasping it at `grasp`.

        Returns a dict with `pose` (4x4, camera frame), `waypoints` and the
        clearance achieved, or None when there is nowhere on the surface that
        fits it. None is a real answer — it means this object cannot be moved
        usefully and a different one should be chosen.

        The object is released in the orientation it was picked up in, so the
        place pose is the grasp pose translated. `keep_clear` names the instance
        that must not end up obstructed again; pass the target's id.

        Examples
        --------
        >>> # Move the bottle out of the banana's way
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     bottle = image_patch.find_by_id("obj_003")
        >>>     grasp = image_patch.grasp_detection(bottle)
        >>>     place = image_patch.place_detection(bottle, grasp, keep_clear="obj_001")
        >>>     return {"grasp": grasp, "place": place}
        """
        import placement as pl

        if grasp is None or not isinstance(grasp, dict) or "pose" not in grasp:
            logger.warning("place_detection: no usable grasp to place from")
            return None
        if REGISTRY is None or REGISTRY.plane is None:
            logger.warning("place_detection: needs a scene registry with a fitted plane")
            return None

        mask = getattr(object_patch, "mask", None)
        if mask is None:
            logger.warning("place_detection: the object patch has no mask")
            return None

        gripper = gripper_name or GRIPPER_NAME
        cloud = unproject(SCENE.depth, SCENE.intrinsics, mask=np.asarray(mask).astype(bool))
        if len(cloud) < MIN_OBJECT_POINTS:
            logger.warning("place_detection: only %d object points", len(cloud))
            return None

        keep_out = None
        if keep_clear is not None and keep_clear in REGISTRY.instances:
            moving = getattr(object_patch, "name", None)
            keep_out = REGISTRY.keep_out_for(
                keep_clear, moving_id=moving if moving in REGISTRY.instances else None
            )

        place = pl.plan_place(
            np.asarray(grasp["pose"], dtype=np.float64),
            cloud,
            REGISTRY.scene_cloud_excluding(),
            REGISTRY.plane,
            gripper=gripper,
            width=float(grasp.get("width", 0.08)),
            keep_out=keep_out,
            hmap=REGISTRY.hmap,
        )
        if place is None:
            logger.info("place_detection: no free space fits this object")
            return None

        logger.info("place_detection: %s", place.summary())
        return place.as_dict()

    @staticmethod
    def _gripper_geometry(gripper_name: str) -> Tuple[float, float]:
        """(jaw aperture, base->fingertip depth) in metres, from the gripper config.

        Both matter for where the grasp actually *lands*: the pose is anchored at
        the gripper base, so the fingertips sit `fingertip_depth` further along
        +Z. Testing mask containment at the base instead of the fingertips checks
        a point ~11 cm behind the contact for a Panda, which rejects good grasps.

        Read from `gripper_descriptions` so the numbers match what the server
        conditioned on; falls back to Franka-like values, which then only affect
        projection and drawing, never the returned pose.

        Delegates to `collision.gripper_geometry`. It used to open `config.json`
        itself, which is how the overlay came to draw every gripper at the
        Franka's fingertip depth while this read the real one.
        """
        return gripper_geometry(gripper_name)

    @classmethod
    def _gripper_jaw_width(cls, gripper_name: str) -> float:
        """Jaw aperture only. Kept as a named accessor for callers/tests."""
        return cls._gripper_geometry(gripper_name)[0]

    # def get_original_boxes(self):
    #     [[dog_patch.left,image_patch.height-dog_patch.upper,dog_patch.right,image_patch.height-dog_patch.lower]]


    # def get_mask(self, object_name: str, box: list):
    #     """Returns a mask of the object in question
    #     Parameters
    #     -------
    #     object name : str
    #         A string describing the name of the object to be masked in the image.
    #     object name : list
    #         Optional list of bounding box values of object.
    #     >>> # Generate the mask of the kid raising their hand
    #     >>> def execute_command(image) -> str:
    #     >>>     image_patch = ImagePatch(image)
    #     >>>     kid_patches = image_patch.find("kid")
    #     >>>     for kid_patch in kid_patches:
    #     >>>         if kid_patch.verify_property("kid", "raised hand"):
    #     >>>             return image_patch.get_mask("kid",[[kid_patch.left,image_patch.height-kid_patch.upper,kid_patch.right,image_patch.height-kid_patch.lower]])
    #     >>>     return None
    #     """
    #     return masks