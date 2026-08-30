CODE = '''

** Role **: You are an expert software programmer. Your task is to convert the given step-by-step instructin plan into executable Python code.
Ensure your code follow the plan. 

** Important instructions **:
1. You will be given the instructions to identify a feasible grasp pose using RGB input. Please use the class python below to write the python code base on the instructions.
2. Your primary responsibility is to translate instructions into Python code. This code will aid in obtaining more visual perception information and conducting logical analysis to arrive at the final answer for query.
3. Image patch is a crop of an image centered around a particular object.
4. You can use base Python (comparison) for basic logical operations, math, etc.

Provided Python Functions/Class:

import math
class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image. Higher values are closer to the right.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
        Measured from the BOTTOM of the image, so higher values are closer to the top.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image. Higher values are closer to the right.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.
        Measured from the BOTTOM of the image, so higher values are closer to the top.
    vertical_center: int
        An int describing the vertical center of the crop's bounding box in the original image.
        Measured from the BOTTOM of the image: higher values are closer to the top. This is NOT an
        image row index. So the HIGHEST object in the picture is sorted(..., key=lambda p: p.vertical_center)[-1]
        and the LOWEST is [0].
    horizontal_center: int
        An int describing the horizontal center of the crop's bounding box in the original image. Higher values are closer to the right.
    mask : array_like
        A boolean segmentation mask of the object (or object part) this patch refers to, at full image
        resolution. This is what grasp_detection uses to decide which pixels become the 3D point cloud.

    Methods
    -------
    find(object_name: str)->list[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
    compute_depth()->float
        Returns the median distance from the camera to this patch, in METRES. Smaller means closer.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    find_part(object_name: str, part_name: str)->ImagePatch
        Returns a new ImagePatch object containing crops of the image centered around a part of an object (object_name) in the image that
        matching the part_name 
    llm_query(question: str)->str
        Returns the answer to a question asked about the image using the LLM model. Typical use when the question is complex, ambiguous, or requires external knowledge. 
        Typically ask about the object properties, relationships between them. For example: Ask the color of the Kleenex package in the image.
    grasp_detection(object_patch: ImagePatch, gripper_name: str = None)->dict
        Return the best 6-DoF grasp pose for the given object_patch (which carries the mask of the
        object or object part). Returns a dict with 3D position in metres, approach and closing
        direction vectors, a confidence score, and the gripper jaw width. Returns None on failure.
        The dict ALSO carries "candidates": every other grasp that survived the same filters,
        best first, each with "index", "score", "position", "approach" and "closing". There are
        typically 15-60 of them and the runners-up score within a few hundredths of the winner.
    select_grasp(grasp: dict, candidates: list)->dict
        Pick a different grasp from the ones already found. Filter grasp["candidates"] however the
        situation calls for, pass what remains, and the best of them comes back as a full grasp.
        Costs nothing — no new detection, no grasp server. Returns None if the list is empty.
        Use this when a grasp has to CHANGE: re-running grasp_detection on the same object returns
        the same pose, because it is the same computation on the same mask.
    find_by_id(object_id: str)->ImagePatch
        Return the object with a given stable instance id, e.g. "obj_003". Use this instead of
        find() whenever the plan names an id: find() cannot tell two identical objects apart.
        Only available while clearing a cluttered table.
    place_detection(object_patch: ImagePatch, grasp: dict, keep_clear: str = None)->dict
        Return where to set an object down after grasping it. Needed whenever the plan says to
        move something out of the way, since a grasp alone cannot relocate anything. Returns None
        when the surface has no room for it.
    """

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # return the foo
        >>> def execute_command(image) -> list[ImagePatch]:
        >>>    image_patch = ImagePatch(image)
        >>>    foo_patches = image_patch.find("foo")
        >>>    return foo_patches

        >>> # Generate the mask of the green book
        >>> def execute_command(image) -> str:
        >>>    image_patch = ImagePatch(image)
        >>>    book_patches = image_patch.find("book")
        >>>    for book_patch in book_patches:
        >>>        if book_patch.verify_property("book", "green"):
        >>>            return book_patch.mask
        
        >>> # Which orange is the leftmost
        >>> def execute_command(image) -> str:
        >>>    image_patch = ImagePatch(image)
        >>>    orange_patches = image_patch.find("orange")
        >>>    orange_patches.sort(key=lambda x: x.left)
        >>>    return orange_patches[0].left, orange_patches[0].lower, orange_patches[0].right, orange_patches[0].upper
        """
        return find_in_image(self.cropped_image, object_name)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, else False.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Grasp the apple if there is one, otherwise grasp the orange
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     if image_patch.exists("apple"):
        >>>         target_patches = image_patch.find("apple")
        >>>     else:
        >>>         target_patches = image_patch.find("orange")
        >>>     return image_patch.grasp_detection(target_patches[0])
        """
        return exists(self.cropped_image, object_name)

    def verify_property(self, object_name: str, visual_property: str) -> bool:
        """Returns True if the object possesses the visual property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property : str
            A string describing the simple visual property (e.g., color, shape, material) to be checked.

        Examples
        -------
        >>> # Do the letters have blue color?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     letters_patches = image_patch.find("letters")
        >>>     # Question assumes only one letter patch
        >>>     return image_patch.bool_to_yesno(letters_patches[0].verify_property("letters", "blue"))
        """
        return verify_property(self.cropped_image, object_name, property)

    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # Which kind of baz is not fredding?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     baz_patches = image_patch.find("baz")
        >>>     for baz_patch in baz_patches:
        >>>         if not baz_patch.verify_property("baz", "fredding"):
        >>>             return baz_patch.simple_query("What is this baz?")

        >>> # What color is the foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patch = foo_patches[0]
        >>>     return foo_patch.simple_query("What is the color?")

        >>> # Is the second bar from the left quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch.simple_query("Is the bar quuxy?")
           
        """
        return simple_query(self.cropped_image, question)

    def crop(self, left: int, lower: int, right: int, upper: int, mask) -> ImagePatch:
        """Returns a new ImagePatch cropped from the current ImagePatch.
        Parameters
        -------
        left, lower, right, upper : int
            The (left/lower/right/upper)most pixel of the cropped image.
        mask
            A mask of the the most prominent object in the crop region. 
        """
        return ImagePatch(self.cropped_image, left, lower, right, upper, mask)

    def best_image_match(list_patches: list[ImagePatch], content: list[str], return_index=False) -> Union[ImagePatch, int]:
        """Returns the patch most likely to contain the content.
        Parameters
        ----------
        list_patches : list[ImagePatch]
        content : list[str]
            the object of interest
        return_index : bool
            if True, returns the index of the patch most likely to contain the object

        Returns
        -------
        int
            Patch most likely to contain the object
        """
        return best_image_match(list_patches, content, return_index)
        
    def compute_depth(self):
        """Returns the median distance from the camera to this patch, in METRES.

        Smaller values are CLOSER to the camera, larger values are FURTHER away.

        Parameters
        ----------
        Returns
        -------
        float
            the median metric depth of the patch, in metres

        Examples
        --------
        >>> # the bar furthest away  (largest depth)
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda bar: bar.compute_depth())
        >>>     return bar_patches[-1]
        >>>
        >>> # the bar closest to the camera  (smallest depth)
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda bar: bar.compute_depth())
        >>>     return bar_patches[0]
        """
        return compute_depth(self)
        
    def find_part(self, object_name: str, part_name: str) -> ImagePatch:
        """Returns a new ImagePatch object containing crops of the image centered around a part of an object (object_name) in the image that
        matching the part_name

        Parameters
        ----------
        object_name : str
            the object of interest
        part_name : str
            the part of the object of interest

        Returns
        -------
        ImagePatch
            ImagePatch of the part of the object of interest

        Examples
        --------
        >>> # Find the blade of the knife
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     knife_patches = image_patch.find("knife")
        >>>     knife_blade_patch = knife_patch[0].find_part("knife", "blade")
        >>>     return knife_blade_patch
        >>>
        >>> # Return the mask of the handle of the spoon
        >>> def execute_command(image)->ImagePatch
        >>>     image_patch = ImagePatch(image)
        >>>     spoon_patches = image_patch.find("spoon")
        >>>     spoon_handle_patch = spoon_patch[0].find_part("spoon", "handle")
        >>>     return spoon_handle_patch.mask
        """
        return find_part(object_name, part_name)
        
    def find_by_id(self, object_id: str) -> ImagePatch:
        """Returns the object the plan named, by its stable instance id.

        Only available while clearing a cluttered table. When the plan names an id like
        "obj_003", use this instead of find(): find() re-runs detection and returns a list
        whose order is not stable, so with two bottles on the table "the bottle" can mean a
        different object each time. An id always means the same physical object.

        Parameters
        ----------
        object_id : str
            an instance id from the plan, e.g. "obj_003"

        Returns
        -------
        ImagePatch
            the patch for that instance

        Examples
        --------
        >>> # Grasp the object the plan named
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     bottle = image_patch.find_by_id("obj_003")
        >>>     return image_patch.grasp_detection(bottle)
        """
        return ImagePatch(image)

    def place_detection(object_patch: ImagePatch, grasp: dict, keep_clear: str = None)->dict:
        """Returns where to put an object down after grasping it.

        Use this whenever the plan says to move an object out of the way. A grasp alone
        cannot relocate anything: the pick and the place are both needed.

        The object is released in the orientation it was picked up in, so only its position
        changes. The place location is chosen to be flat, empty, big enough for the object,
        and clear of the target named by keep_clear - otherwise clearing an object could
        mean setting it back down in front of the thing you were trying to reach.

        Parameters
        ----------
        object_patch : ImagePatch
            the object being moved
        grasp : dict
            the grasp returned by grasp_detection for that object
        keep_clear : str, optional
            the instance id that must not end up obstructed again, i.e. the target

        Returns
        -------
        dict
            A dict with these keys, or None if there is nowhere to put the object:
              "pose"        - 4x4 release pose (camera frame, metres)
              "position"    - [x, y, z] release position in metres
              "travel_m"    - how far the object moves
              "clearance_m" - free space around the chosen spot
              "waypoints"   - the poses to move through, in order

        None is a real answer. It means the surface has no room for this object, and a
        different object should be moved instead.

        Examples
        --------
        >>> # Move the bottle out of the banana's way
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     bottle = image_patch.find_by_id("obj_003")
        >>>     grasp = image_patch.grasp_detection(bottle)
        >>>     place = image_patch.place_detection(bottle, grasp, keep_clear="obj_001")
        >>>     return {{"grasp": grasp, "place": place}}
        """
        return {{}}

    def grasp_detection(object_patch: ImagePatch, gripper_name: str = None)->dict:
        """Returns the best 6-DoF grasp pose for the object/part in the object_patch.

        This is a full 3D robot hand pose, not a 2D rectangle. The approach direction is free:
        the model may propose grasping from the side, from above, or at an angle, whichever fits
        the object's 3D geometry.

        The mask carried by object_patch determines which part of the scene is lifted into 3D,
        so passing a part patch from find_part() produces a grasp on that part specifically.

        Parameters
        ----------
        object_patch : ImagePatch
            the object or object part to grasp
        gripper_name : str, optional
            the robot gripper to plan for, e.g. "franka_panda", "robotiq_2f_85", "unitree_g1".
            Only pass this if the user explicitly asks for a particular robot or gripper.

        Returns
        -------
        dict
            A dict with these keys (or None if no valid grasp was found):
              "pose"     - 4x4 homogeneous transform (camera frame, metres)
              "position" - [x, y, z] gripper position in metres
              "approach" - [x, y, z] unit vector the gripper advances along
              "closing"  - [x, y, z] unit vector the fingers close along
              "score"    - grasp confidence in [0, 1]
              "width"    - jaw opening in metres
              "gripper"  - the gripper name used
              "rect_2d"  - [score, x, y, w, h, angle] projection into the image

        Examples
        --------
        >>> # Return the grasp pose of the object
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     grasp_pose = image_patch.grasp_detection(object_patch)
        >>>     return grasp_pose
        >>>
        >>> # Grasp the plant at its pot
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     plant_patches = image_patch.find("plant")
        >>>     pot_patches = plant_patches[0].find_part("plant", "pot")
        >>>     grasp_pose = image_patch.grasp_detection(pot_patches[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the orange on the plate
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     orange_patches = image_patch.find("orange")
        >>>     plate_patches = image_patch.find("plate")[0]
        >>>     oranges_on_plate = [
        >>>         orange for orange in orange_patches 
        >>>        if ( orange.vertical_center > plate_patches.lower and orange.vertical_center < plate_patches.upper and 
        >>>        orange.horizontal_center > plate_patches.left and orange.horizontal_center < plate_patches.right )
        >>>     ]
        >>>     if len(oranges_on_plate) == 0:
        >>>         return None
        >>>     grasp_pose = image_patch.grasp_detection(oranges_on_plate[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the furthest object
        >>> # compute_depth() is metric distance in metres, so SMALLER is CLOSER
        >>> # and the furthest object is the one with the LARGEST depth.
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     object_patches.sort(key=lambda object: object.compute_depth())
        >>>     grasp_pose = image_patch.grasp_detection(object_patches[-1])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the nearest object
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     object_patches.sort(key=lambda object: object.compute_depth())
        >>>     grasp_pose = image_patch.grasp_detection(object_patches[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the knife next to the fork
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     fork_patches = image_patch.find("fork")
        >>>     knife_patches = image_patch.find("knife")
        >>>     knife_patches.sort(key=lambda knife: abs(knife.horizontal_center - fork_patches[0].horizontal_center))
        >>>     grasp_pose = image_patch.grasp_detection(knife_patches[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the knife by its handle (part-level grasp)
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     knife_patches = image_patch.find("knife")
        >>>     handle_patch = knife_patches[0].find_part("knife", "handle")
        >>>     grasp_pose = image_patch.grasp_detection(handle_patch)
        >>>     return grasp_pose
        >>>
        >>> # Grasp the mug with the Robotiq gripper
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     mug_patches = image_patch.find("mug")
        >>>     grasp_pose = image_patch.grasp_detection(mug_patches[0], gripper_name="robotiq_2f_85")
        >>>     return grasp_pose
        """
        return grasp_detection(object_patch, gripper_name)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left, lower, right, upper : int
            the (left/lower/right/upper) border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False

        Examples
        --------
        >>> # black foo on top of the qux
        >>> def execute_command(image) -> ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str) -> str:
        """Returns the answer to a question asked about the image using the LLM model. Typical use when the question is complex, ambiguous, or requires external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        
        Returns
        -------
        str

        Examples
        -------
        >>> # What is the color of the Kleenex package in the image?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     return image_patch.llm_query("What is the color of the Kleenex package in the image?")
        """
        return llm_query(question)

Write a function using Python and the ImagePatch class (above) that could be executed to provide an answer to the query. 


### Examples
{example}

Plan at this step: {plan}
** Expected format output begin with **
def execute_command(image):
'''

EXAMPLES_CODER = '''
### Example 1
Plan:
Step 1: Find the carrot in the image.
Step 2: Detect the grasp pose for the first detected carrot.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    carrot_patches = image_patch.find("carrot")
    grasp_pose = image_patch.grasp_detection(carrot_patches[0])
    return grasp_pose
    ```

### Example 2
Plan:
Step 1: Find all patches containing bottles in the image.
Step 2: Iterate through each detected bottle patch. 
Step 3: Verify if the bottle is both blue and red.
Step 4: Perform grasp pose detection for the blue bottle.
Step 5: Handle the case where no blue bottles are found. Return None.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bottle_patches = image_patch.find("bottle")
    for bottle_patch in bottle_patches:
        if bottle_patch.verify_property("bottle", "blue and red"):
                grasp_pose = image_patch.grasp_detection(bottle_patch)
                return grasp_pose
    return None
    ```

### Example 3
Plan:
Step 1: Find all patches containing chocolate bars in the image.
Step 2: Sort the chocolate bar patches based on their horizontal position.
Step 3: The second chocolate bar from the left will be the second element in the sorted list.
Step 4: Return the grasp pose for the second chocolate bar from the left.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bar_patches = image_patch.find("chocolate bar")
    bar_patches.sort(key=lambda x: x.horizontal_center)
    bar_patch = bar_patches[1]
    grasp_pose = image_patch.grasp_detection(bar_patch)
    return grasp_pose
    ```

### Example 4
Plan:
Step 1: Find all apples in the image.
Step 2: Sort the apples patches based on their vertical position.
Step 3: The apple at highest position is the last item in the list.
Step 4: Return the grasp pose for the highest position apple.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    apple_patches = image_patch.find("apple")
    apple_patches.sort(key=lambda x: x.vertical_center)
    apple_patch = apple_patches[-1]
    grasp_pose = image_patch.grasp_detection(apple_patch)
    return grasp_pose
    ```

### Example 5
Plan: 
Step 1: Detect all knives in the image.
Step 2: The knife is to be handed to the person, so the robot must hold the "blade" and present the handle. Locate the "blade" part of the first detected knife.
Step 3: Calculate the grasp pose for the knife blade.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    knife_patches = image_patch.find("knife")
    knife_patch = knife_patches[0]
    knife_blade_patch = knife_patch.find_part("knife", "blade")
    grasp_pose = image_patch.grasp_detection(knife_blade_patch)
    return grasp_pose
    ```

### Example 5b
Plan: 
Step 1: Find obj_004 in the image.
Step 2: obj_004 is a knife being moved out of the way, not handed over, so take the most secure hold. Locate its "handle" part.
Step 3: Calculate the 6-DoF grasp pose for the handle of obj_004.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    obj_004 = image_patch.find_by_id("obj_004")
    handle_patch = obj_004.find_part("knife", "handle")
    grasp_pose = image_patch.grasp_detection(handle_patch)
    return grasp_pose
    ```

### Example 5c
Plan: 
Step 1: Find obj_002 in the image.
Step 2: Compute the grasp pose for obj_002.
Step 3: The previous attempt was rejected because the hand approached through the object beside it, which lies to the +X side. Choose a grasp that approaches from a different direction instead.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    obj_002 = image_patch.find_by_id("obj_002")
    grasp_pose = image_patch.grasp_detection(obj_002)
    if grasp_pose is None:
        return None
    # Keep only the alternatives that do not come in from the +X side.
    alternatives = [c for c in grasp_pose["candidates"] if c["approach"][0] < 0.2]
    better = image_patch.select_grasp(grasp_pose, alternatives)
    return better if better is not None else grasp_pose
    ```

### Example 6
Plan: 
Step 1: Question about the Kleenex box in the image, find out its color.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    kleenex_info = image_patch.llm_query("What is the color of the Kleenex package in the image?")
    return kleenex_info
    ```
'''