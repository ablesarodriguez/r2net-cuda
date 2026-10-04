import cupy as cp
import numpy as np
from utils.padding import apply_edge_padding, crop_to_original

@cp.fuse()
def applyactfunc_cuda(z):
    #This function applies a fused clipped Sigmoid activation on the GPU to avoid latency
    #Input arguments:
    #z: batch of pre-activation values
    
    #Output:
    #Activated batch tensor normalized to [0,1]
    
    #Clip values to prevent overflow in np.exp
    z = cp.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + cp.exp(-z))

def R2Net_predict_batch_cuda(gt_batch, x_batch, learning_rate, W, b, er_p):
    #This function implements the R2Net to predict an entire diagonal batch of blocks on the GPU
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
        
        #Native CUDA 3D Matrix Multiplication
        z = cp.matmul(cp.transpose(W, (0, 2, 1)), x_batch) + cp.transpose(b, (0, 2, 1))
        hz = applyactfunc_cuda(z)
        pred = er_p + hz

        #Backward propagation to FC layer
        er = (gt_batch - pred) if it == 0 else (pred - gt_batch)
        
        dW = -(cp.matmul(x_batch, cp.transpose(er, (0, 2, 1)))) if it == 0 else \
              cp.matmul(x_batch, cp.transpose(er, (0, 2, 1))) #Opposite direction for gradient
        
        #Gradient descent
        W -= lr * dW
        b -= lr * cp.transpose(er, (0, 2, 1)) #gradient for b
        
        er_p = er #update error
        
    return pred

def R2Net_reconstruct_batch_cuda(x_batch, res_batch, learning_rate, W, b, er_p):
    #This function implements the R2Net to reconstruct a diagonal batch of blocks on the GPU
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
    
    #Dynamic epsilon based on precision
    EPS = dtype.type(1e-15) if dtype == cp.float64 else dtype.type(1e-7)
    
    #Apply limits to avoid log(0) and perform inverse activation (logit)
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

def setup_context_mapping_cuda(block_size, FIXED_INPUT_SIZE, dtype):
    #This function generates the context mapping matrix and uploads it to the GPU
    #Input arguments:
    #block_size: size of processing blocks
    #FIXED_INPUT_SIZE: target size for the output context vector
    #dtype: data type for the matrices

    #Output:
    #M: CuPy transformation matrix to apply mean downsampling via matrix multiplication
    #valid_groups: CuPy boolean mask indicating valid context vector positions
    
    L = 2 * block_size + 1
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
# 3. PIPELINES WAVEFRONT CUDA (COMPRESIÓN Y DESCOMPRESIÓN)
# ================================================================================

def compress_wf_cuda(img, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16):
    #This function compresses an image using a GPU wavefront (diagonal) batch processing approach
    #Input arguments:
    #img: original image to compress (NumPy array)
    #block_size: size of processing blocks
    #learning_rate: learning rate for prediction
    #FIXED_INPUT_SIZE: size of context feature vector

    #Output:
    #error_map_gpu: compressed representation containing the residuals (returned as NumPy array)
    
    img_padded_np = apply_edge_padding(img, block_size)
    dtype = img_padded_np.dtype
    
    #Transfer padded image to GPU
    img_gpu = cp.asarray(img_padded_np)
    h, w = img_gpu.shape
    b = block_size
    
    error_map_gpu = cp.zeros_like(img_gpu)
    # 1px border buffer for neighbors
    padded_rec = cp.full((h + 1, w + 1), 0.5, dtype=dtype)
    rows_b, cols_b = h // b, w // b
    
    M_gpu, valid_groups_gpu = setup_context_mapping_cuda(b, FIXED_INPUT_SIZE, dtype)

    # Pre-allocation of memory buffers to avoid latency spikes during iterations
    max_N = min(rows_b, cols_b)
    W_buf = cp.empty((max_N, FIXED_INPUT_SIZE, b*b), dtype=dtype)
    b_buf = cp.empty((max_N, 1, b*b), dtype=dtype)
    er_buf = cp.empty((max_N, b*b, 1), dtype=dtype)
    b_range = cp.arange(b)

    for k in range(rows_b + cols_b - 1):
        # 1. Identify blocks belonging to diagonal 'k'
        start_r, end_r = max(0, k - cols_b + 1), min(rows_b - 1, k)
        N = end_r - start_r + 1
        rs = cp.arange(start_r, end_r + 1)
        cs = k - rs
        py, px = rs * b, cs * b
        
        # 2. Extract Ground Truth
        r_grid = py[:, None, None] + b_range[None, :, None]
        c_grid = px[:, None, None] + b_range[None, None, :]
        gt_batch = img_gpu[r_grid, c_grid].reshape(N, b * b, 1)
        
        # 3. Extract Context L
        left_p = padded_rec[py[:, None] + 1 + b_range, px[:, None]]
        tl_p = padded_rec[py[:, None], px[:, None]]
        top_p = padded_rec[py[:, None], px[:, None] + 1 + b_range]
        all_p = cp.concatenate([left_p, tl_p, top_p], axis=1)
        
        # 4. Matrix-based Downsampling (Vectorized)
        x_batch = cp.matmul(all_p, M_gpu).reshape(N, FIXED_INPUT_SIZE, 1)
        x_batch[:, ~valid_groups_gpu] = 0.5 #Padding value
        
        # 5. Calculation and Reconstruction using pre-allocated buffers
        W_curr, b_curr, er_curr = W_buf[:N], b_buf[:N], er_buf[:N]
        pred_batch = R2Net_predict_batch_cuda(gt_batch, x_batch, learning_rate, W_curr, b_curr, er_curr)
        err_batch = gt_batch - pred_batch
        rec_batch = R2Net_reconstruct_batch_cuda(x_batch, err_batch, learning_rate, W_curr, b_curr, er_curr)
        
        # 6. Dump results into the error map and reconstruction buffer
        error_map_gpu[r_grid, c_grid] = err_batch.reshape(N, b, b)
        padded_rec[r_grid + 1, c_grid + 1] = rec_batch.reshape(N, b, b)

    #Transfer back to CPU
    return cp.asnumpy(error_map_gpu)

def decompress_wf_cuda(error_map, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16, original_shape=None):
    #This function decompresses an image using a GPU wavefront (diagonal) batch processing approach
    #Input arguments:
    #error_map: compressed representation containing residuals (NumPy array)
    #block_size: size of processing blocks
    #learning_rate: learning rate for reconstruction
    #FIXED_INPUT_SIZE: size of context feature vector
    #original_shape: shape of the image prior to padding

    #Output:
    #Cropped reconstructed image (returned as NumPy array)
    
    dtype = error_map.dtype
    
    #Transfer error map to GPU
    error_map_gpu = cp.asarray(error_map)
    h, w = error_map_gpu.shape
    b = block_size
    
    # 1px border buffer for neighbors
    padded_rec = cp.full((h + 1, w + 1), 0.5, dtype=dtype)
    rows_b, cols_b = h // b, w // b
    
    M_gpu, valid_groups_gpu = setup_context_mapping_cuda(b, FIXED_INPUT_SIZE, dtype)

    # Pre-allocation of memory buffers to avoid latency spikes during iterations
    max_N = min(rows_b, cols_b)
    W_buf = cp.empty((max_N, FIXED_INPUT_SIZE, b*b), dtype=dtype)
    b_buf = cp.empty((max_N, 1, b*b), dtype=dtype)
    er_buf = cp.empty((max_N, b*b, 1), dtype=dtype)
    b_range = cp.arange(b)

    for k in range(rows_b + cols_b - 1):
        # 1. Identify blocks belonging to diagonal 'k'
        start_r, end_r = max(0, k - cols_b + 1), min(rows_b - 1, k)
        N = end_r - start_r + 1
        rs = cp.arange(start_r, end_r + 1)
        cs = k - rs
        py, px = rs * b, cs * b
        
        # 2. Extract batch of residuals
        r_grid = py[:, None, None] + b_range[None, :, None]
        c_grid = px[:, None, None] + b_range[None, None, :]
        res_batch = error_map_gpu[r_grid, c_grid].reshape(N, b * b, 1)
        
        # 3. Extract Context L from previously reconstructed blocks
        left_p = padded_rec[py[:, None] + 1 + b_range, px[:, None]]
        tl_p = padded_rec[py[:, None], px[:, None]]
        top_p = padded_rec[py[:, None], px[:, None] + 1 + b_range]
        all_p = cp.concatenate([left_p, tl_p, top_p], axis=1)
        
        # 4. Matrix-based Downsampling (Vectorized)
        x_batch = cp.matmul(all_p, M_gpu).reshape(N, FIXED_INPUT_SIZE, 1)
        x_batch[:, ~valid_groups_gpu] = 0.5 #Padding value
        
        # 5. Reconstruction using pre-allocated buffers
        W_curr, b_curr, er_curr = W_buf[:N], b_buf[:N], er_buf[:N]
        rec_batch = R2Net_reconstruct_batch_cuda(x_batch, res_batch, learning_rate, W_curr, b_curr, er_curr)
        
        # 6. Update reconstruction buffer
        padded_rec[r_grid + 1, c_grid + 1] = rec_batch.reshape(N, b, b)

    #Transfer back to CPU and crop to original size
    rec_cpu = cp.asnumpy(padded_rec[1:, 1:])
    return crop_to_original(rec_cpu, original_shape)