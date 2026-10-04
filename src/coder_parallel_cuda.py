import numpy as np
import cupy as cp
from utils.padding import apply_edge_padding, crop_to_original

# ================================================================================
# 1. KERNELS CUDA EXTREMOS (Fusión y Cero-Asignación)
# ================================================================================

@cp.fuse()
def applyactfunc_cuda(z):
    #This function applies a fused clipped Sigmoid activation on the GPU
    #Input arguments:
    #z: batch of pre-activation values
    
    #Output:
    #Activated batch tensor normalized to [0,1]
    
    #Fuses clip, exp, and division into a single step on the GPU chip
    z = cp.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + cp.exp(-z))

def R2Net_predict_batch_cuda(gt_batch, x_batch, learning_rate, W, b, er_p):
    #This function implements the R2Net to predict a batch of blocks on the GPU using pre-allocated buffers
    #Input arguments:
    #gt_batch: batch of ground truth blocks, organized as 1-column vectors, normalized to [0,1]
    #x_batch: batch of feature vectors for prediction, organized as 1-column vectors, normalized to [0,1]
    #learning_rate: learning rate
    #W: pre-allocated weight tensor for the batch
    #b: pre-allocated bias tensor for the batch
    #er_p: pre-allocated error tensor for the batch

    #Output:
    #pred: predicted batch of blocks, organized as 1-column vectors, normalized to [0,1]
    
    #Reset pre-allocated buffers to their initial values
    W.fill(1.0)
    b.fill(1.0)
    er_p.fill(0.0)
    lr = W.dtype.type(learning_rate)

    for it in (0, 1): #Two iterations of back-propagation (BP)
        #Forward propagation
        
        #Batch Matrix Multiplication
        z = cp.matmul(cp.transpose(W, (0, 2, 1)), x_batch) + cp.transpose(b, (0, 2, 1))
        hz = applyactfunc_cuda(z)
        pred = er_p + hz

        #Backward propagation to FC layer
        er = (gt_batch - pred) if it == 0 else (pred - gt_batch)
        
        dW = -(cp.matmul(x_batch, cp.transpose(er, (0, 2, 1)))) if it == 0 else \
              cp.matmul(x_batch, cp.transpose(er, (0, 2, 1)))
        
        #Gradient descent
        W -= lr * dW
        b -= lr * cp.transpose(er, (0, 2, 1)) #gradient for b
        
        er_p = er #update error
        
    return pred

def R2Net_reconstruct_batch_cuda(x_batch, res_batch, learning_rate, W, b, er_p):
    #This function implements the R2Net to reconstruct a batch of blocks on the GPU using pre-allocated buffers
    #Input arguments:
    #x_batch: batch of feature vectors, organized as 1-column vectors, normalized to [0,1]
    #res_batch: batch of residuals (error signals) of the blocks
    #learning_rate: learning rate
    #W: pre-allocated weight tensor for the batch
    #b: pre-allocated bias tensor for the batch
    #er_p: pre-allocated error tensor for the batch

    #Output:
    #Reconstructed batch of blocks, organized as 1-column vectors, clipped to [0,1]
    
    #Reset pre-allocated buffers to their initial values
    W.fill(1.0)
    b.fill(1.0)
    er_p.fill(0.0)
    dtype = W.dtype
    lr = dtype.type(learning_rate)
    
    #Forward propagation to obtain initial prediction f1
    z = cp.matmul(cp.transpose(W, (0, 2, 1)), x_batch) + cp.transpose(b, (0, 2, 1))
    f1 = applyactfunc_cuda(z)
    
    #Apply limits to avoid log(0) and perform inverse activation (logit)
    EPS = dtype.type(1e-15) if dtype == cp.float64 else dtype.type(1e-7)
    
    f2 = cp.clip(-res_batch + er_p + f1, EPS, dtype.type(1.0) - EPS)
    f2i = cp.log(f2) - cp.log(dtype.type(1.0) - f2)
    
    #Compute analytical gradient step for reconstruction
    scalar = cp.matmul(cp.transpose(x_batch, (0, 2, 1)), x_batch)
    denom = (lr * scalar) - lr
    #Prevent division by zero by safely bounding the denominator
    denom = cp.where(cp.abs(denom) < EPS, EPS * cp.sign(denom + dtype.type(1e-16)), denom)

    er1 = (f2i - z) / denom
    
    #Return reconstructed signal clipped to valid image range
    return cp.clip(er1 + er_p + f1, dtype.type(0.0), dtype.type(1.0))

# ================================================================================
# 2. GESTIÓN DE CONTEXTO VECTORIZADA (CUDA)
# ================================================================================

def setup_context_mapping(block_size, FIXED_INPUT_SIZE, dtype):
    #This function precomputes a mapping matrix to downsample context pixels for a GPU batch
    #Input arguments:
    #block_size: size of processing blocks
    #FIXED_INPUT_SIZE: target size for the output context vector
    #dtype: data type for the matrices

    #Output:
    #M_gpu: CuPy transformation matrix to apply mean downsampling via matrix multiplication
    #valid_groups_gpu: CuPy boolean mask indicating valid context vector positions
    
    L = (2 * block_size) - 1
    K = FIXED_INPUT_SIZE
    splits = np.array_split(np.zeros(L), K)
    
    M = np.zeros((L, K), dtype=dtype)
    valid_groups = np.zeros(K, dtype=bool)
    
    idx = 0
    for i, g in enumerate(splits):
        s = len(g)
        if s > 0:
            M[idx:idx+s, i] = dtype.type(1.0) / dtype.type(s) #Set mean weights
            idx += s
            valid_groups[i] = True
            
    #Transfer initialized matrices to GPU
    return cp.asarray(M), cp.asarray(valid_groups)

# ================================================================================
# 3. PIPELINE PARALLEL CUDA (CORREGIDO)
# ================================================================================

def compress_parallel_cuda(img, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16):
    #This function compresses an entire image in parallel on the GPU
    #Input arguments:
    #img: original image to compress (NumPy array)
    #block_size: size of processing blocks
    #learning_rate: learning rate for prediction
    #FIXED_INPUT_SIZE: size of context feature vector

    #Output:
    #error_map_gpu: compressed representation containing the residuals (returned as NumPy array)
    
    img_padded = apply_edge_padding(img, block_size)
    dtype = img_padded.dtype
    h, w = img_padded.shape
    b = block_size
    
    #Transfer padded image to GPU
    img_gpu = cp.asarray(img_padded, dtype=dtype)
    
    # 1. Block slicing
    blocks = img_gpu.reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b, b)
    N = blocks.shape[0] # Total blocks
    
    # 2. Extract DPCM references
    top_row, left_col = blocks[:, 0, :], blocks[:, :, 0]
    
    dpcm_top = cp.zeros_like(top_row)
    dpcm_top[:, 0] = top_row[:, 0]
    dpcm_top[:, 1:] = top_row[:, 1:] - top_row[:, :-1]
    
    dpcm_left = cp.zeros_like(left_col)
    dpcm_left[:, 0] = left_col[:, 0]
    dpcm_left[:, 1:] = left_col[:, 1:] - left_col[:, :-1]
    
    rec_top, rec_left = cp.cumsum(dpcm_top, axis=1), cp.cumsum(dpcm_left, axis=1)
    
    # 3. Context preparation
    all_pixels = cp.concatenate([rec_left, rec_top[:, 1:]], axis=1)
    M_gpu, valid_groups_gpu = setup_context_mapping(b, FIXED_INPUT_SIZE, dtype)
    
    x_batch = cp.matmul(all_pixels, M_gpu)
    x_batch[:, ~valid_groups_gpu] = dtype.type(0.5) #Padding value
    x_batch = x_batch.reshape(N, FIXED_INPUT_SIZE, 1)
    
    # 4. PRE-ALLOCATION (The key to performance)
    sb, sfv = b * b, FIXED_INPUT_SIZE
    W_buf = cp.empty((N, sfv, sb), dtype=dtype)
    b_buf = cp.empty((N, 1, sb), dtype=dtype)
    er_p_buf = cp.empty((N, sb, 1), dtype=dtype) # Initial er_p
    
    gt_batch = blocks.reshape(N, sb, 1)
    
    # CORRECTED CALL: Passing the buffers
    pred_batch = R2Net_predict_batch_cuda(gt_batch, x_batch, learning_rate, W_buf, b_buf, er_p_buf)
    err_batch = gt_batch - pred_batch
    
    # 5. DPCM injection and final map assembly
    err_blocks = err_batch.reshape(N, b, b)
    err_blocks[:, 0, :] = dpcm_top
    err_blocks[:, :, 0] = dpcm_left
    
    error_map_gpu = err_blocks.reshape(h // b, w // b, b, b).transpose(0, 2, 1, 3).reshape(h, w)
    
    #Transfer back to CPU
    return cp.asnumpy(error_map_gpu)


def decompress_parallel_cuda(error_map, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16, original_shape=None):
    #This function decompresses an image in parallel on the GPU
    #Input arguments:
    #error_map: compressed representation containing residuals (NumPy array)
    #block_size: size of processing blocks
    #learning_rate: learning rate for reconstruction
    #FIXED_INPUT_SIZE: size of context feature vector
    #original_shape: shape of the image prior to padding

    #Output:
    #Cropped reconstructed image (returned as NumPy array)
    
    dtype = error_map.dtype
    h, w = error_map.shape
    b = block_size
    
    #Transfer error map to GPU and slice
    err_gpu = cp.asarray(error_map, dtype=dtype)
    err_blocks = err_gpu.reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b, b)
    N = err_blocks.shape[0]
    
    # 1. Vectorized DPCM Reconstruction
    dpcm_top, dpcm_left = err_blocks[:, 0, :], err_blocks[:, :, 0]
    rec_top, rec_left = cp.cumsum(dpcm_top, axis=1), cp.cumsum(dpcm_left, axis=1)
    
    # 2. Context mapping
    all_pixels = cp.concatenate([rec_left, rec_top[:, 1:]], axis=1)
    M_gpu, valid_groups_gpu = setup_context_mapping(b, FIXED_INPUT_SIZE, dtype)
    
    x_batch = cp.matmul(all_pixels, M_gpu)
    x_batch[:, ~valid_groups_gpu] = dtype.type(0.5) #Padding value
    x_batch = x_batch.reshape(N, FIXED_INPUT_SIZE, 1)
    
    # 3. PRE-ALLOCATION
    sb, sfv = b * b, FIXED_INPUT_SIZE
    W_buf = cp.empty((N, sfv, sb), dtype=dtype)
    b_buf = cp.empty((N, 1, sb), dtype=dtype)
    er_p_buf = cp.empty((N, sb, 1), dtype=dtype)
    
    res_batch = err_blocks.reshape(N, sb, 1)
    
    # 4. Passing the buffers
    rec_batch = R2Net_reconstruct_batch_cuda(x_batch, res_batch, learning_rate, W_buf, b_buf, er_p_buf)
    
    # 5. Final Assembly
    rec_blocks = rec_batch.reshape(N, b, b)
    rec_blocks[:, 0, :] = rec_top
    rec_blocks[:, :, 0] = rec_left
    
    rec_img_gpu = rec_blocks.reshape(h // b, w // b, b, b).transpose(0, 2, 1, 3).reshape(h, w)
    
    #Transfer back to CPU and crop to original size
    return crop_to_original(cp.asnumpy(rec_img_gpu), original_shape)