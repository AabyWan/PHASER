import os, pathlib
from PIL import Image, ImageDraw, ImageEnhance
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List, Tuple

from ..utils import ImageLoader


class Transformer(ABC):
    # Abstract class for transformers. Specifies the interfaces to use when defining a transformer.
    def __init__(
        self, name: str = "", saveToDir: str = "", saveToSubDir: bool = True
    ) -> None:
        """Abstract constructor for Transformer class to pass through common properties.

        Args:
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.
        """
        self.name = name
        self.saveToDir = saveToDir
        self.saveToSubDir = saveToSubDir

    @abstractmethod
    def fit(self) -> Image.Image:
        """Abstract method that should be defined for all transforms to actually create the modified image. Return a PIL.Image object.

        Returns:
            Image: A tranformed copy of the original image.
        """
        pass

    def saveToDisk(self, image: Image.Image, save_directory:str, filename: str) -> str:
        """Save the transformed file to disk. Should only be called when needed.

        Args:
            image (Image): PIL.Image object corresponding to the transform.
            original_path (str): Path of the original file, only used to get the filename for saving the new file (using the same file type)

        Returns:
            filepath (str): The path the file was saved to.
        """

        if self.saveToSubDir:
            dirpath = os.path.join(save_directory, self.name)
        else:
            dirpath = save_directory
        pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(os.path.join(dirpath, filename))
        try:
            image.save(filepath)
        except OSError as e:
            if "RGBA" in str(e):
                rgb_im = image.convert('RGB')
                rgb_im.save(filepath)
        return(filepath)


# Special case, get transform from disk
class TransformFromDisk(Transformer):
    """Special case Transformer. Rather than create a new object from an original image, load one that already exists from disk.
    Does not require a save path or the subdirectory flag, as it shouldn't need to write anything to disk. Must call the fit function to apply the transform and return a PIL.Image.

    Args:
        Transformer (_type_): Base class inheritence, though it's not really needed in this case.
    """

    def __init__(self, dirpath: str, name: str = "") -> None:
        """Initialise the transform by specifying the path to the folder containing the file. Filename is automatically derived from the image being "transformed", so they need to match.

        Args:
            dirpath (str): The path of the directory containing the target transforms. The appropriate filename will be used to load the transform, so it needs to have the same name as the original it is being matched to.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".

        Raises:
            Exception: Raise a generic exception if the target directory doesn't exist.

        Returns:
            None: call fit to return the modified PIL.Image.
        """
        self.dirpath = dirpath
        if not os.path.isdir(self.dirpath):
            raise Exception(
                f"Invalid path provided to TransformFromDisk: {self.dirpath}. Expected a directory."
            )

        if name:
            self.name = name
        else:
            # Use the folder name if no name is provided.
            self.name = os.path.split(self.dirpath)[-1]

    def fit(self, image_obj) -> Image.Image:
        filename = image_obj.filename
        path = os.path.join(self.dirpath, filename)
        image_obj = ImageLoader(path=path)

        return image_obj.image  # image from disk

class Blend(Transformer):
    def __init__(
        self,
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
        direction = "Left-to-Right",
        static_image: str = r"../resources/blend_static.jpg",
    ) -> None:
        """Blend the source image with a static image, specified by static_image. direction specifies where to begin the fade-in for the static image. Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.
            direction (str, optional): Set the blend direction for the static image. Blending Left-to-Right means the static blend image fades in from the left (low-opacity), with higher opacity on the right. Possible options: 'left-to-right', 'right-to-left', 'up-to-down', 'down-to-up'
            static_image(str, optional): Path to the image to create a composite with the target image. Defaults to r"../resources/blend_static.jng".

        Returns:
            None: call fit to return the modified PIL.Image.
        """

        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = f"Blend_{direction}"
    
        if static_image == r"../resources/blend_static.jpg":
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self.static_image = os.path.join(
                package_dir, static_image
            )
        else:
            self.static_image = static_image
        # Check blend direction and set
        directions = ['left-to-right', 'right-to-left', 'up-to-down', 'down-to-up']
        if direction.lower() in directions:
            self.direction = direction
        else:
            raise(Exception(f"Invalid direction {direction} specified in Blend transform construction."))
              
        # Load the blending image to use
        self.blend_im = Image.open(self.static_image)

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        # resize the static image to match the target
        resized_blend_im = self.blend_im.resize((image.width, image.height))
        
        if image.size != resized_blend_im.size:
            raise ValueError("HorzintalComposite: Both images must have the same dimensions.")
        
        alpha_mask = Image.new('L', (image.width, image.height))
        
        if self.direction in ['left-to-right', 'right-to-left']:
            # Draw the gradiant across the x-axis
            for x in range(image.width):
                # Calculate the alpha value based on the position
                alpha_value = (int((x / image.width) * 255))
                # Draw a vertical line with the calculated alpha value
                if self.direction == 'left-to-right':
                    ImageDraw.Draw(alpha_mask).line([(x, 0), (x, image.height)], fill=alpha_value)
                else:
                    ImageDraw.Draw(alpha_mask).line([(x, 0), (x, image.height)], fill=255-alpha_value)
        else:
            # Vertical instead
            # Draw the gradiant across the y-axis
            for y in range(image.height):
                # Calculate the alpha value based on the position
                alpha_value = (int((y / image.height) * 255))
                # Draw a horizontal line with the calculated alpha value
                if self.direction == 'up-to-down':
                    ImageDraw.Draw(alpha_mask).line([(0, y), (image.width, y)], fill=alpha_value)
                else:
                    ImageDraw.Draw(alpha_mask).line([(image.width, y), (0, y)], fill=255-alpha_value)

        image = Image.composite(image, resized_blend_im, alpha_mask)
    

        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image

class Border(Transformer):
    def __init__(
        self,
        border_colour: Tuple[int, int, int] = (255, 255, 255),
        border_width: int = 10,
        border_width_fraction: float = 0.0,
        saveToDir: str = "",
        name: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """Create a transform which adds a border (specify width and RGB colour) to an image. Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            border_colour (Tuple[int, int, int], optional):  An integer triple of RGB values for the colour of the border.. Defaults to (255,255,255).
            border_width (int, optional): The border width (in pixels) to add to the image. Defaults to 10.
            border_width_fraction (float, optional): The border width as a fraction of the image width (as a percentage) to add to the image. Defaults to 0%. Overrides border_width.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.
        """
        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = (
                f"Border_bw{border_width}_bc{'.'.join([str(n) for n in border_colour])}"
            )
        self.bc = border_colour
        self.bw = border_width
        self.bwf = border_width_fraction

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)
        
        # Set border width if fraction is specified
        if int(self.bwf) != 0:
            self.bw = int(image.width * (self.bwf / 100))
            
        # Draw the rectangle border on the image
        canvas = ImageDraw.Draw(image)
        canvas.rectangle(
            [(0, 0), (image.width - 1, image.height - 1)],  # type:ignore
            outline=self.bc,
            width=self.bw,
        )

        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image

class Composite(Transformer):
    def __init__(
        self,
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
        position = "left",
        coords: Tuple[int] = (), # pixel location to paste
        scale: bool = False,
        scale_size: Tuple[int] = (), # resize tuple for static image
        static_image: str = r"../resources/blend_static.jpg"
    ) -> None:
        """Insert the static image, specified by static_image, to create a composite image. The position of the static image is defined by the position argument. Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. 
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.
            position (str, optional): Set the position of the static image when embedding. Possible options: 'left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right'.Defaults to bottom-left. Defaults to 'left'.
            coords (Tuple[int], optional): Manually specify where to paste the image, otherwise this is calculated automatically by the position.
            scale (bool, optional): Select whether to scale the static image or not. The image is scaled to 1/2 (left/right) or 1/4 (quadrants) if True, otherwise it is only scaled to the size of the original and will 'show through' as if a piece of the original was cut out.
            scale_size (Tuple[int], optional): Manually specify scaling for the static image.
            static_image(str, optional): Path to the image to create a composite with the target image. Defaults to r"../resources/blend_static.jng".

        Returns:
            None: call fit to return the modified PIL.Image.
        """

        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            if coords != ():
                self.name = f"Composite_static_c{coords}"
            else:
                self.name = f"Composite_static_p{position}"
    
            # Manage relative paths
        try:
            thisdir = os.path.dirname(__file__)
            if static_image == r"../resources/blend_static.jpg":
                self.static_image = os.path.join(thisdir, static_image)
            else:
                self.static_image = static_image
        except Exception as e:
            print(e)
            print(f"""Problem encoutered loading image to embed for Composite Transform. 
                  Check path: {static_image}""")
        
    
        # if static_image == r"../resources/blend_static.jpg":
        #     package_dir = os.path.dirname(os.path.abspath(__file__))
        #     self.static_image = os.path.join(
        #         package_dir, static_image
        #     )
        # else:
        #     self.static_image = static_image
            
        if coords != ():
            # Check position and set
            positions = ['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right']
            if not position.lower() in positions:
                raise(Exception(f"Invalid position {position} specified in Composite transform construction."))
        self.position = position.lower()
        self.scale = scale
        self.scale_size = scale_size
        self.coords = coords
        
        # Load the blending image to use
        self.static_im = Image.open(self.static_image)
    
    def set_insert_coords(self, position: str, image_width: int, image_height: int) -> None:
        formula_dict = {
            'left': (0, 0),
            'right': (int(image_width / 2),  0), 
            'top': (0, 0),  
            'bottom': (0, int(image_height / 2)), 
            'top-left': (0, 0), 
            'top-right': (int(image_width / 2),  0), 
            'bottom-left': (0, int(image_height / 2)), 
            'bottom-right': (int(image_width / 2), int(image_height / 2))
        }
        self.coords = formula_dict[position]

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        
        # Perform image scaling
        
        if self.scale_size:
            # Specify a scaling for the static image
            resized_static_im = self.static_im.resize(self.scale_size)
        else:
            if self.scale == True:
                # Dynamic scaling based on position
                if self.position in ['left', 'right']:
                    resized_static_im = self.static_im.resize((int(image.width / 2), image.height))
                elif self.position in ['top', 'bottom']:
                    resized_static_im = self.static_im.resize((image.width, int(image.height / 2)))
                elif self.position in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
                    resized_static_im = self.static_im.resize((int(image.width / 2), int(image.height / 2)))
            else:
                # resize the static image to match the target
                resized_static_im = self.static_im.resize((image.width, image.height))

        # Paste the static image
        if not self.coords:
            self.set_insert_coords(self.position, image.width, image.height)
        image.paste(resized_static_im, self.coords)
        

        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Crop(Transformer):
    """Class to crop an image based on a box specifying how much to remove from each edge. Specified as a list of either int (factors to multiple dimension by before removing)
    or a list of integers specifying exact number of pixels to crop. Specified as (left, upper, right, lower) in the list.
    Example: Cropping by cropbox_absolute=[5,10,10,5] removes 5 pixels drom the left and bottom edges, and 10 pixels from the top and right edge.
    cropbox_factors=[0.1, 0.2, 0.1, 0.2] removes 10% of the x-axis length from both the left and right, and 20% of the y-axis length from top and bottom. Must call the fit function to apply the transform and return a PIL.Image.

    Args:
        Transformer (class): The base abstract class to inherit methods from.
    """

    def __init__(
        self,
        cropbox_factors: List[float] = [],
        cropbox_absolute: List[int] = [],
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """Create a Crop transform.

        Args:
            cropbox_factors (list[float]): The crop rectangle as a list of factors (a multiple of the dimension), specified as (left, upper, right, lower) in the list.
            cropbox_absolute (list[int]): The crop rectangle as a list of pixel dimensions to snip from each edge (fixed size), specified as (left, upper, right, lower) in the list.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.

        """
        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if cropbox_factors and cropbox_absolute:
            raise Exception(
                "Transformer.Rescale requires either cropbox_factors or cropbox_absolute, not both."
            )
        elif cropbox_factors:
            self.cropbox_factors = cropbox_factors
            for f in self.cropbox_factors:
                if not(f >= 0.0 and f <=1.0):
                    raise Exception(f"Incorrect crop-factors provided. Should be a list of floats between 0.0 and 1.0. Got {self.cropbox_factors}")
            if not self.name:
                self.name = f"Crop_factors{str(cropbox_factors)}"
            self.cropbox_absolute = None
        elif cropbox_absolute:
            self.cropbox_absolute = cropbox_absolute
            if not self.name:
                self.name = f"Crop_fixed{str(cropbox_absolute)}"
            self.cropbox_factors = None

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        if self.cropbox_factors:
            cropbox = (
                int(image.width * self.cropbox_factors[0]),
                int(image.height * self.cropbox_factors[1]),
                int(image.width * (1 - self.cropbox_factors[2])),
                int(image.height * (1 - self.cropbox_factors[3])),
            )
        else:
            cropbox = (
                int(self.cropbox_absolute[0]),
                int(self.cropbox_absolute[1]),
                int(image.width - self.cropbox_absolute[2]),
                int(image.height - self.cropbox_absolute[3]),
            )
        image = image.crop(cropbox)


        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Enhance(Transformer):
    def __init__(
        self,
        brightnessfactor: float = 1.0,
        colourfactor: float = 1.0,
        contrastfactor: float = 1.0,
        sharpnessfactor: float = 1.0,
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """Perform colour, brightness, contrast and sharpness enhancements to the image. Any arguments which aren't 1 will be processed in order: colour, contrast, brightness, sharpness.
        Leverages PIL.ImageEnhance, which has details, but, essentially: 1.0 is the original image. For each factor, 0.0 does: brightness: black image, colour: black and white image,
        contrast: solid grey image, sharpness: blurred image (2.0 for sharpening). Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            brightnessfactor (float, optional): _description_. Defaults to 1.0.
            colourfactor (float, optional): _description_. Defaults to 1.0.
            contrastfactor (float, optional): _description_. Defaults to 1.0.
            sharpnessfactor (float, optional): _description_. Defaults to 1.0.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.
        """
        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = f"Enhance_colour{colourfactor}bright{brightnessfactor}contrast{contrastfactor}sharp{sharpnessfactor}"
        self.colourfactor = colourfactor
        self.brightnessfactor = brightnessfactor
        self.contrastfactor = contrastfactor
        self.sharpnessfactor = sharpnessfactor

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        if self.colourfactor != 1.0:
            current_colour = ImageEnhance.Color(image)
            image = current_colour.enhance(self.colourfactor)
        if self.brightnessfactor != 1.0:
            current_contrast = ImageEnhance.Contrast(image)
            image = current_contrast.enhance(self.contrastfactor)
        if self.contrastfactor != 1.0:
            current_brightness = ImageEnhance.Brightness(image)
            image = current_brightness.enhance(self.brightnessfactor)
        if self.sharpnessfactor != 1.0:
            current_sharpness = ImageEnhance.Sharpness(image)
            image = current_sharpness.enhance(self.sharpnessfactor)
       
       

        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Flip(Transformer):
    def __init__(
        self,
        direction="horizontal",
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """Flip the image on the x (horizontal) or y (verical) axis. Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            direction (str, optional): Flip axis to apply to the image. horizontal or vertical. Defaults to "horizontal".
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.
        """

        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = f"Flip_{direction}"
        self.direction = direction.lower()

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        if self.direction == "horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)


        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Rescale(Transformer):
    def __init__(
        self,
        scalefactor: float = None,
        fixed_dimensions: Tuple[int, int] = None,
        thumbnail_aspect: bool = True,
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """_summary_ Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            scalefactor (float): A scaling factor for the image (incompatible with fixed_dimensions). Multiplies dimensions by this factor and rounds down to nearest pixel.
            fixed_dimensions (tuple): A tuple for (x,y) fixed pixel dimensions to rescale the image to.
            thumbnail_aspect (bool, optional): Flag to mantain aspect ratio when changing the overall dimesions - typical thumbnail behaviour. If specifying fixed_dimensions, forces those dimensions to act as a maximum, rather than absolute values. Defaults to False.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.
        Raises:
            Exception: Scaling factor and fixed dimensions are incompatible, raise generic exception.

        Returns:
            None: call fit to return the modified PIL.Image.
        """
        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)

        if scalefactor and fixed_dimensions:
            raise Exception(
                "Transformer.Rescale requires either scalefactor or fixed_dimensions, not both."
            )
        elif scalefactor:
            self.scalefactor = scalefactor
            if not self.name:
                self.name = f"Rescale{str(scalefactor)}x"
            self.fixed_dimensions = None
        elif fixed_dimensions:
            self.dimensions = fixed_dimensions
            if not self.name:
                self.name = f"Rescale_fixed{str(fixed_dimensions)}"
            self.scalefactor = None

        self.thumbnail = thumbnail_aspect

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        if self.scalefactor:
            self.dimensions = (
                int(image.width * self.scalefactor),
                int(image.height * self.scalefactor),
            )

        if self.thumbnail:
            # Thumbnail method preserves aspect ratio when scaling, may not match exact size spcified by fixed_dimensions.
            # Method modifies data in place, for some reason, contrary to most other methods.
            image.thumbnail(self.dimensions)
        else:
            # Resize, returns modified data.
            image = image.resize(self.dimensions)


        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Rotate(Transformer):
    def __init__(
        self,
        degrees_counter_clockwise: int = 5,
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """_summary_ Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            degrees_counter_clockwise (int, optional): Specifty the rotation value in degrees. Positive values rotate counter-clockwise, negative values rotate clockwise. Defaults to 5.
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.
        """
        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = f"Rotate_{str(degrees_counter_clockwise)}"
        self.degrees = degrees_counter_clockwise

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)
        image = image.rotate(self.degrees)


        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image


class Watermark(Transformer):
    def __init__(
        self,
        image_height_factor: float = 0.1,
        minheight: int = 40,
        watermark_path: str = r"../resources/watermark.png",
        name: str = "",
        saveToDir: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """_summary_ Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            image_height_factor (float, optional): If supplied, scales the watermark to this decimal fraction of the original image's height. Defaults to 0.1.
            minheight (int, optional): The minimum height of the watermark to attempt to enforce. Gets scaled down if too big for the target image. Defaults to 40.
            watermark_path (str, optional): Path to the watermark image to embed Transparency is preserved if possible. Defaults to r"../resources/watermark.png".
            name (str, optional): Name for the transform - optional, otherwise derived from the provided arguments. Serves as directory name if saveToSubDir is set to True, otherwise can be used in graphs. Defaults to "".
            saveToDir (str, optional): Output directory if provided, if not, then the transformed image is not saved. If saveToSubDir is set then this will be the top-level directory. Defaults to ''.
            saveToSubDir (bool, optional): Requires saveToDir to be set to do anything - creates a subdirectory based on the name of the transform. Defaults to False.

        Returns:
            None: call fit to return the modified PIL.Image.
        """

        super().__init__(name=name, saveToDir=saveToDir, saveToSubDir=saveToSubdir)
        if not self.name:
            self.name = f"Watermark"
        self.height_factor = image_height_factor
        self.minheight = minheight

        if watermark_path == r"../resources/watermark.png":
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self.watermark_path = os.path.join(
                package_dir, r"../resources/watermark.png"
            )
        # Load the watermark image to use
        self.watermark_im = Image.open(self.watermark_path)

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

        targetheight = int(image.height * self.height_factor)
        waterwidthscaler = self.watermark_im.width / self.watermark_im.height
        targetwidth = int(targetheight * waterwidthscaler)

        # Enforce minimum size
        if targetheight < self.minheight:
            targetheight = self.minheight
            targetwidth = int(targetheight * waterwidthscaler)

            # It's too wide, reduce to fit
            if targetwidth > image.width:
                targetwidth = int(image.width)

        # Scale the watermark, keep aspect ratio
        self.watermark_im.convert("RGBA")
        self.watermark_im.thumbnail((targetwidth, targetheight))

        # Add the watermark to the image with transparency.
        self.watermark_im.convert("RGBA")
        image.paste(
            self.watermark_im,
            (image.width - targetwidth, image.height - targetheight),
            self.watermark_im,
        )


        if self.saveToDir:
            self.saveToDisk(image=image, save_directory=self.saveToDir, filename=image_obj.filename)

        return image
