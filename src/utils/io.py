import numpy as np
from pathlib import Path

def parse_encoding(encoding_str):
    # This function parses a string to determine bits, endianness and sign.
    
    # Input arguments:
    # encoding_str: String describing the encoding (e.g., 'u8', 'sle16', 'sbe32')
    
    # Output:
    # bits: Integer representing the number of bits (e.g., 8, 16)
    # endianness_char: Character for endianness ('<' for little, '>' for big)
    # signed_char: Character for data type ('u' for unsigned, 'i' for signed)

    encoding_lower = encoding_str.lower()
    bits = 0
    endianness_char = '<'  # Default to Little-endian
    signed_char = 'u'      # Default to Unsigned

    if 'ube' in encoding_lower:
        endianness_char = '>'
    elif 'sbe' in encoding_lower:
        endianness_char = '>'
        signed_char = 'i'  # 'i' for signed integer
    elif 'ule' in encoding_lower:
        endianness_char = '<'

    # Support for 'sle' if needed
    elif 'sle' in encoding_lower:
        endianness_char = '<'
        signed_char = 'i'

    if '16' in encoding_lower or '2' in encoding_lower:
        bits = 16
    elif '8' in encoding_lower or '1' in encoding_lower:
        bits = 8

    if bits == 0:
        raise ValueError(f"Could not determine number of bits from encoding: '{encoding_str}'")

    return bits, endianness_char, signed_char


def load_raw_image(filepath, width, height, channels, encoding):
    # This function loads a single RAW image from a file with specific encoding handling.
    
    # Input arguments:
    # filepath: Path to the raw image file
    # width: Width of the image
    # height: Height of the image
    # channels: Number of color channels (1 for grayscale)
    # encoding: Encoding string (e.g., 'u8', 'ube16')
    
    # Output:
    # dict: Dictionary containing the numpy array and metadata parameters, or None if failed.

    try:
        bits, endianness_char, signed_char = parse_encoding(encoding)
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            raw_data = f.read()

        # Construct the numpy dtype string
        dtype_str = f'{endianness_char}{signed_char}{bits // 8}'
        image_array = np.frombuffer(raw_data, dtype=np.dtype(dtype_str))
        
        expected_pixels = width * height * channels
        if image_array.size != expected_pixels:
            print(f"Warning: File size ({image_array.size} px) does not match expected size ({expected_pixels} px). Data will be truncated.")
            image_array = image_array[:expected_pixels]

        shape = (height, width, channels) if channels > 1 else (height, width)
        image_array = image_array.reshape(shape)

        return {
            'array': image_array,
            'params': {
                'width': width, 
                'height': height, 
                'channels': channels,
                'bits': bits, 
                'endianness': 'big' if endianness_char == '>' else 'little',
                'encoding': encoding
            }
        }
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' does not exist.")
        return None
    except Exception as e:
        print(f"Error processing '{filepath}': {e}")
        return None


def save_raw_image(image_array, filepath):
    # This function saves a numpy array as a RAW image file.
    
    # Input arguments:
    # image_array: Numpy array containing the image data
    # filepath: Destination path for the file
    
    # Output:
    # bool: True if saved successfully, False otherwise

    try:
        output_path = Path(filepath)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_array.tofile(output_path)
        print(f"Image successfully saved to '{output_path}'")
        return True
    except Exception as e:
        print(f"Error saving image to '{filepath}': {e}")
        return False