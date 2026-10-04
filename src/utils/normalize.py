import numpy as np

def normalize_min_max_values(data, dtype=np.float64):
    data_typed = data.astype(dtype)
    
    min_val = np.min(data_typed)
    max_val = np.max(data_typed)
    
    # Protección: si el rango es cero (color sólido), devolver zeros
    if max_val == min_val:
        return min_val, max_val, np.zeros_like(data_typed)
    
    normalized = (data_typed - min_val) / (max_val - min_val)
    return min_val, max_val, normalized


def denormalize_min_max_values(normalized_data, original_min, original_max):
    # This function reverses the normalization to restore original values.
    
    # Input arguments:
    # normalized_data: The array with values between 0 and 1.
    # original_min: The minimum value of the data before it was normalized.
    # original_max: The maximum value of the data before it was normalized.
    
    # Use the input array's dtype to avoid type promotion
    dtype = normalized_data.dtype
    
    # Restore the float values using the same precision as the input
    restored_float = normalized_data * (dtype.type(original_max) - dtype.type(original_min)) + dtype.type(original_min)
    restored_rounded = np.round(restored_float)
 
    # Return as integer to match the original image format (uint8/uint16)
    return restored_rounded.astype(int)

def normalize_minus1_to_1(data):
    #This function normalizes a numpy array to the range [-1, 1]
    #Input arguments:
    #data: input numpy array to be normalized

    #Output:
    #min_val: minimum value of the original data
    #max_val: maximum value of the original data
    #data_minus1_1: normalized data in range [-1, 1]
    
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    
    #Protection: If the block is solid color (max == min), avoid division by zero
    if range_val == 0:
        #If all values are equal, return zeros (center of range -1 to 1)
        return min_val, max_val, np.zeros_like(data, dtype=float)
    
    #Normalize to [0, 1]
    data_01 = (data - min_val) / range_val
    
    #Scale to [-1, 1] using formula (x * 2) - 1
    data_minus1_1 = (data_01 * 2) - 1
    
    return min_val, max_val, data_minus1_1

def denormalize_minus1_to_1(normalized_data, original_min, original_max):
    #This function reverts the normalization from [-1, 1] to the original range
    #Input arguments:
    #normalized_data: data in range [-1, 1]
    #original_min: original minimum value
    #original_max: original maximum value

    #Output:
    #restored_rounded: denormalized data, rounded and cast to int
    
    range_val = original_max - original_min
    
    if range_val == 0:
        #If the original range was 0, return array filled with min value
        return np.full_like(normalized_data, original_min).astype(int)

    #Revert to [0, 1] using formula (x + 1) / 2
    data_01 = (normalized_data + 1) / 2
    
    #Revert to the original range
    restored_float = data_01 * range_val + original_min
    
    #Round and convert to integer
    #Use clip for safety against floating point errors
    restored_rounded = np.round(restored_float)
    
    return restored_rounded.astype(int)