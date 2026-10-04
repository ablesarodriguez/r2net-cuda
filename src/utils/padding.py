import numpy as np

def apply_edge_padding(img, block_size):
    """
    Applies 'edge padding' to an image if its dimensions are not multiples of the block_size.
    This ensures the image can be perfectly divided into blocks.
    
    Inputs:
    - img (numpy.ndarray): The 2D image array to pad.
    - block_size (int): The size of the blocks to be processed (e.g., 16).
    
    Outputs:
    - numpy.ndarray: The padded image, or the original image if no padding is needed.
    """
    h_orig, w_orig = img.shape
    pad_h = (block_size - (h_orig % block_size)) % block_size
    pad_w = (block_size - (w_orig % block_size)) % block_size
    
    if pad_h > 0 or pad_w > 0:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    return img

def crop_to_original(img, original_shape):
    """
    Crops a reconstructed image back to its original dimensions, removing any padding.
    
    Inputs:
    - img (numpy.ndarray): The reconstructed image array (potentially with padding).
    - original_shape (tuple or None): The original shape (height, width) of the image.
    
    Outputs:
    - numpy.ndarray: The cropped image matching the original shape.
    """
    if original_shape is not None:
        return img[:original_shape[0], :original_shape[1]]
    return img