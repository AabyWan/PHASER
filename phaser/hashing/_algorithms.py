from abc import ABC, abstractmethod
import imagehash
import pdqhash
import numpy as np
import os
import onnxruntime # neuralhash


class PerceptualHash(ABC):
    """ Abstract class to extend when implementing hashing algorithms. The constructor is used to pass through arguments in fit,
    which ultimately calls the underlying library/implementation.
    """
    @abstractmethod
    def __init__(self):
        """Function to initialise the hash function, pass in any parameters here."""

    @abstractmethod
    def fit(self, img):
        """Function that takes in an image and returns a hash digest as a bit string."""


class AverageHash(PerceptualHash):
    """average_hash implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/"""
    def __init__(self, hash_size=8, mean_func=np.mean):
        self.hash_size = hash_size
        self.mean_func = mean_func

    def fit(self, img):
        ihash = imagehash.average_hash(
            image=img, hash_size=self.hash_size, mean=self.mean_func
        ).hash

        ihash = np.array(ihash).flatten()
        return ihash


class ColourHash(PerceptualHash):
    """HSV Color Hash implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/"""
    def __init__(self, binbits=3) -> None:
        self.binbits = binbits

    def fit(self, img) -> str:
        ihash = imagehash.colorhash(image=img, binbits=self.binbits).hash

        flat_hash = np.concatenate(ihash).flatten()
        return flat_hash
    
# uses a different hash output setup, need to think about how to integrate it.
# class CropResistantHash(PerceptualHash):
#     """Crop Resistant Hah (rHash) implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/
#     M. Steinebach, H. Liu and Y. Yannikos, "Efficient Cropping-Resistant Robust Image Hashing," 2014 Ninth International Conference on Availability, Reliability and Security, Fribourg, Switzerland, 2014, pp. 579-585, doi: 10.1109/ARES.2014.85.
#     Needs to be handled differently to the other hashes in ImageHash as it returns multi-part hashes.
#     """

#     def fit(self, img) -> str:
#         ihash = imagehash.crop_resistant_hash(image=img).hash

#         flat_hash = np.concatenate(ihash).flatten()
#         return flat_hash
    
class DifferenceHash(PerceptualHash):
    """Difference Hash (dhash) implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/"""
    def __init__(self, hash_size=8, vertical=False) -> None:
        self.hash_size = hash_size
        self.vertical = vertical
    def fit(self, img) -> str:
        if self.vertical:
            ihash = imagehash.dhash_vertical(image=img, hash_size=self.hash_size).hash
        else:
            ihash = imagehash.dhash(image=img, hash_size=self.hash_size).hash

        flat_hash = np.concatenate(ihash).flatten()
        return flat_hash

    
class NeuralHash(PerceptualHash):
    """
    Apple Neuralhash, implementation from https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX.
    Model and DAT files must be extracted from an appropriate iOS image.
    """
    
    def __init__(self, model_file=r"../resources/model.onnx", 
                 dat_file=r"../resources/neuralhash_128x96_seed1.dat",
                 pad=0):
        
        # Manage relative paths
        try:
            thisdir = os.path.dirname(__file__)
            if model_file == r"../resources/model.onnx":
                self.model_file = os.path.join(thisdir, model_file)
            else:
                self.model_file = model_file
            if dat_file == r"../resources/neuralhash_128x96_seed1.dat":
                self.dat_file = os.path.join(thisdir, dat_file)
            else:
                self.dat_file = dat_file
        except Exception as e:
            print(e)
            print("""Problem encoutered loading Neuralhash model and dat files. 
                  By default these can be placed in the resources folder. 
                  See https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX for details on how to obtain these.""")
        
        # Load ONNX model
        session = onnxruntime.InferenceSession(self.model_file)

        # Load output hash matrix
        seed1 = open(self.dat_file, 'rb').read()[128:]
        seed1 = np.frombuffer(seed1, dtype=np.float32)
        seed1 = seed1.reshape([96, 128])
        
        self.session = session
        self.seed = seed1
        self.pad = pad

    def fit(self, img):        
        img = img.resize([360, 360])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

        # Run model
        inputs = {self.session.get_inputs()[0].name: arr}
        outs = self.session.run(None, inputs)
        hash_output = self.seed.dot(outs[0].flatten())
        
        # Pad the result if required. 
        if self.pad > 0:
            hash_output = np.pad(hash_output,self.pad, mode="mean")
        nhash = np.array([True if it >= 0 else False for it in hash_output])
        
        return nhash

    
class PHash(PerceptualHash):
    """pHash implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/"""
    def __init__(self, hash_size=8, highfreq_factor=4):
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def fit(self, img):
        ihash = imagehash.phash(
            image=img, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor
        ).hash

        ihash = np.array(ihash).flatten()
        return ihash


class WaveHash(PerceptualHash):
    """Wavelet hash implementation imported from the ImageHash library. https://pypi.org/project/ImageHash/"""
    def __init__(
        self, hash_size=8, image_scale=None, mode="haar", remove_max_haar_ll=True
    ) -> None:
        self.hash_size = hash_size
        self.image_scale = image_scale
        self.mode = mode
        self.remove_max_haar_ll = remove_max_haar_ll

    def fit(self, img) -> str:
        ihash = imagehash.whash(
            image=img,
            hash_size=self.hash_size,
            image_scale=self.image_scale,
            mode=self.mode,
            remove_max_haar_ll=self.remove_max_haar_ll,
        ).hash

        flat_hash = np.concatenate(ihash).flatten()
        return flat_hash



class PdqHash(PerceptualHash):
    """
    PDQ (Facebook) hash implementation from https://github.com/faustomorales/pdqhash-python.
    Based on pHash
    """

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