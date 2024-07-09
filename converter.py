from image_enhancement import image_enhancement
import cv2 as cv
import numpy as np
import pydicom

def normalize_img(img, bits = 8, dtype = np.uint8, eps: float = 1e-9):
    """Normalize an image to a desired bit depth and dtype."""
    # norm to 0-1
    img = (img - np.min(img)) / ((np.max(img) - np.min(img)) + eps)
    # norm to desired bit depth and dtype
    return (img*((2**bits)-1)).astype(dtype)

def apply_method(img, method):
    """Apply a contrast enhancement method to an image."""
    method_dict = {
        'clahe': apply_clahe, 
        'he': apply_he, 
        'apply_bbhe': apply_bbhe, 
        'apply_agcwd': apply_agcwd
    }
    if method is None:
        return img
    elif method in method_dict.keys():
        method_func = method_dict.get(method, None)
        return method_func(img)
    else:
        raise ValueError(f"Invalid method '{method}', choose one of {list(method_dict.keys())}.")
    
def apply_he(img):
    """Apply histogram equalization to an image using the image_enhancement implementation."""
    ie = image_enhancement.IE(img, 'grayscale')
    return ie.GHE()

def apply_clahe(img, clipLimit: float = 5.0):
    """Apply CLAHE to an image using the OpenCV implementation. Defaults to a clipLimit of 5.0."""
    clahe = cv.createCLAHE(clipLimit=clipLimit)
    return clahe.apply(img)[..., np.newaxis]

def apply_bbhe(img):
    """Apply BBHE to an image using the image_enhancement implementation."""
    ie = image_enhancement.IE(img, 'grayscale')
    return ie.BBHE()

def apply_agcwd(img):
    """Apply AGCWD to an image using the image_enhancement implementation."""
    ie = image_enhancement.IE(img, 'grayscale')
    return ie.AGCWD()

def convert_dicom(dcm, method):
    """Convert a DICOM image to a normalized, contrast-enhanced image."""
    img = dcm.pixel_array

    # invert img if necessary based on PhotometricInterpretation value
    if dcm['PhotometricInterpretation'].value == 'MONOCHROME1':
        img = np.max(img) - img

    # pad the image from (h, w) to (h, w, 1)
    img = img[..., np.newaxis]
    
    # normalize the image (preserving bit depth as long as possible)
    img = normalize_img(img, bits=16, dtype=np.uint16)

    # apply contrast enhancement method
    img = apply_method(img, method)

    return img
    

class DataFrameConverter:
    def __init__(self, df, method='clahe', output_paths=True, dcm_path_col='anon_dicom_path'):
        """
        Initialize a DataFrameConverter object with a DataFrame and contrast enhancement method.
        """
        self.df = df
        self.method = method
        self.output_paths = output_paths
        self.dcm_path_col = dcm_path_col

    def __iter__(self):
        """
        Iterate through the DataFrame, converting each DICOM image to a normalized, contrast-enhanced image.
        Yields the image and its corresponding DICOM path.
        """
        for i, data in self.df.iterrows():
            with pydicom.dcmread(data[self.dcm_path_col]) as dcm:
                img = convert_dicom(dcm, self.method)
                yield img, data[self.dcm_path_col]

    def __len__(self):
        return len(self.df)
    

def convert_df(df, method = 'clahe', save=True):
    """
    Convert a DataFrame of DICOM images to a list of normalized, contrast-enhanced images.
    """
    converter = DataFrameConverter(df, method)

    imgs = [img for img in converter]
    return imgs