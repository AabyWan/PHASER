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
        image.save(filepath)
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
                f"Invalid path provided to TransformFromDisk: {self.dirpath}. Expeted a directory."
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


# -------


class Border(Transformer):
    def __init__(
        self,
        border_colour: Tuple[int, int, int] = (255, 255, 255),
        border_width: int = 10,
        saveToDir: str = "",
        name: str = "",
        saveToSubdir: bool = True,
    ) -> None:
        """Create a transform which adds a border (specify width and RGB colour) to an image. Must call the fit function to apply the transform and return a PIL.Image.

        Args:
            border_colour (Tuple[int, int, int], optional):  An integer triple of RGB values for the colour of the border.. Defaults to (255,255,255).
            border_width (int, optional): The border width (in pixels) to add to the image. Defaults to 10.
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

    def fit(self, image_obj) -> Image.Image:
        image = deepcopy(image_obj.image)

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
            if not self.name:
                self.name = f"Crop_{str(cropbox_factors)}"
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
            self.name = f"Ehance_colour{colourfactor}bright{brightnessfactor}contrast{contrastfactor}sharp{sharpnessfactor}"
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
