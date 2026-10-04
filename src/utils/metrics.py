import numpy as np
import math

def calculate_metrics(original_array, processed_array, bits):
    # This function calculates the PAE, MSE, and PSNR metrics between two images.
    
    # Input arguments:
    # original_array: Numpy array of the original image (ground truth).
    # processed_array: Numpy array of the processed/reconstructed image.
    # bits: Bit depth of the original image (e.g., 8 for standard, 16 for medical/raw).
    
    # Output:
    # dict: A dictionary containing 'PAE', 'MSE', and 'PSNR' values.

    # Ensure arrays are float64 for precision during calculation
    original_float = original_array.astype(np.float64)
    processed_float = processed_array.astype(np.float64)
    
    # --- PAE (Peak Absolute Error) ---
    # PAE measures the maximum absolute difference between any two corresponding pixels
    abs_error = np.abs(original_float - processed_float)
    pae = np.max(abs_error)
    
    # --- MSE (Mean Squared Error) ---
    # MSE measures the average of the squares of the errors
    squared_error = (original_float - processed_float) ** 2
    mse = np.mean(squared_error)
    
    # --- PSNR (Peak Signal-to-Noise Ratio) ---
    # PSNR is an expression for the ratio between the maximum possible value (power) of a signal
    # and the power of distorting noise that affects the quality of its representation.
    if mse == 0:
        psnr = float('inf')
    else:
        max_i = (2**bits) - 1
        psnr = 20 * np.log10(max_i / np.sqrt(mse))
        
    return {
        'PAE': pae,
        'MSE': mse,
        'PSNR': psnr
    }

def print_metrics(metrics_dict):
    # This function prints the calculated metrics in a readable format.
    
    # Input arguments:
    # metrics_dict: A dictionary containing 'PAE', 'MSE', and 'PSNR' values.
    
    print(f"Peak Absolute Error (PAE): {metrics_dict['PAE']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics_dict['MSE']:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {metrics_dict['PSNR']:.2f} dB")