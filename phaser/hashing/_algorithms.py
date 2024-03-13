import imagehash
import pdqhash
import numpy as np
from abc import ABC, abstractmethod

# Local imports from ..utils
from ..utils import bool2binstring


class PerceptualHash(ABC):
    @abstractmethod
    def __init__(self):
        """Function to initialise the hash function, pass in any parameters here."""
        pass

    @abstractmethod
    def fit(self):
        """Function that takes in an image and returns a hash digest as a bit string."""
        pass


class PHash(PerceptualHash):
    def __init__(self, hash_size=8, highfreq_factor=4):
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def fit(self, img):
        hash = imagehash.phash(
            image=img, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor
        ).hash

        hash = np.array(hash).flatten()
        return hash
        # Convert bool array to a string
        #binary_hash = bool2binstring(hash)
        #return binary_hash


class ColourHash(PerceptualHash):
    def __init__(self, binbits=3) -> None:
        self.binbits = binbits

    def fit(self, img) -> str:
        hash = imagehash.colorhash(image=img, binbits=self.binbits).hash

        flat_hash = np.concatenate(hash).flatten()
        return flat_hash
        #binary_hash = bool2binstring(flat_hash)
        #return binary_hash


class WaveHash(PerceptualHash):
    def __init__(
        self, hash_size=8, image_scale=None, mode="haar", remove_max_haar_ll=True
    ) -> None:
        self.hash_size = hash_size
        self.image_scale = image_scale
        self.mode = mode
        self.remove_max_haar_ll = remove_max_haar_ll

    def fit(self, img) -> str:
        hash = imagehash.whash(
            image=img,
            hash_size=self.hash_size,
            image_scale=self.image_scale,
            mode=self.mode,
            remove_max_haar_ll=self.remove_max_haar_ll,
        ).hash

        flat_hash = np.concatenate(hash).flatten()
        return flat_hash
        #binary_hash = bool2binstring(flat_hash)
        #return binary_hash


class PdqHash(PerceptualHash):
    def __init__(self) -> None:
        pass

    def fit(self, img):
        # https://github.com/faustomorales/pdqhash-python
        # pip install pdqhash
        # pdq expects the images as a numpy array. Convert accordingly
        # https://stackoverflow.com/questions/384759/how-do-i-convert-a-pil-image-into-a-numpy-array
        # np.bool_(pdqhash.compute(np.asarray(self.image))[0])
        flat_hash = pdqhash.compute(np.asarray(img))[0].astype(bool)
        return flat_hash
        # hex_hash = bool2hex(flat_hash)
        #binary_hash = bool2binstring(flat_hash)
        #return binary_hash
