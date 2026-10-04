import numpy as np
from utils.padding import apply_edge_padding, crop_to_original

def applyactfunc_batch(z):
    #This function applies the Sigmoid activation function to a batch of inputs with a safety limit
    #Input arguments:
    #z: input batch tensor for activation
    
    #Output:
    #result: activated batch values using the Sigmoid function bounded to [0,1]
    
    dtype = z.dtype
    # Fixed and safe clip to prevent overflows
    z = np.clip(z, dtype.type(-500.0), dtype.type(500.0))
    return dtype.type(1.0) / (dtype.type(1.0) + np.exp(-z))

def R2Net_predict_batch(gt_batch, x_batch, learning_rate):
    #This function implements the R2Net to predict a batch of blocks
    #Input arguments:
    #gt_batch: ground truth signals for a batch of blocks, organized as a 3D tensor
    #x_batch: feature vectors of context used for prediction, organized as a 3D tensor
    #learning_rate: learning rate
    
    #Output:
    #pred: predicted batch of blocks, organized as a 3D tensor
    
    dtype = gt_batch.dtype
    batch_size = gt_batch.shape[0]
    sb = gt_batch.shape[1]  # sb = size of gt
    sfv = x_batch.shape[1]  # sfv = size of x
    
    # Initialize weights (W) and bias (b) to one
    W = np.ones((batch_size, sfv, sb), dtype=dtype)
    b = np.ones((batch_size, 1, sb), dtype=dtype)
    er_p = np.zeros((batch_size, sb, 1), dtype=dtype)
    lr = dtype.type(learning_rate)

    # Constant for gradient vanishing protection
    TINY = dtype.type(1e-15) if dtype == np.float64 else dtype.type(1e-7)

    for it in (0, 1): # Two iterations of back-propagation (BP)
        # Forward propagation
        z = np.matmul(np.transpose(W, (0, 2, 1)), x_batch) + np.transpose(b, (0, 2, 1))
        hz = applyactfunc_batch(z)
        pred = er_p + hz
        
        # Backward propagation
        if it == 0:
            er = (gt_batch - pred)
            dW = -(np.matmul(x_batch, np.transpose(er, (0, 2, 1))))
        else:
            er = (pred - gt_batch)
            dW = np.matmul(x_batch, np.transpose(er, (0, 2, 1)))
        
        # Protection against vanishing gradients
        dW = np.where(np.abs(dW) < TINY, TINY * np.sign(dW + TINY), dW)
        
        # Gradient descent (weight update)
        W -= lr * dW
        b -= lr * np.transpose(er, (0, 2, 1))
        er_p = er 
        
    return pred

def R2Net_reconstruct_batch(x_batch, res_batch, learning_rate):
    #This function implements the R2Net to reconstruct a batch of blocks from their error signals
    #Input arguments:
    #x_batch: feature vectors of context used for prediction, organized as a 3D tensor
    #res_batch: error signals or residuals for a batch of blocks, organized as a 3D tensor
    #learning_rate: learning rate
    
    #Output:
    #rec: reconstructed batch of blocks bounded to [0,1], organized as a 3D tensor
    
    dtype = res_batch.dtype
    lr = dtype.type(learning_rate)
    
    batch_size = res_batch.shape[0]
    sb = res_batch.shape[1]  # sb = size of res
    sfv = x_batch.shape[1]   # sfv = size of x
    
    # Initialize weights (W) and bias (b) to one
    W = np.ones((batch_size, sfv, sb), dtype=dtype)
    b = np.ones((batch_size, 1, sb), dtype=dtype)
    er_p = np.zeros((batch_size, sb, 1), dtype=dtype)
    
    # Forward propagation to obtain the initial prediction f1
    z = np.matmul(np.transpose(W, (0, 2, 1)), x_batch) + np.transpose(b, (0, 2, 1))
    f1 = applyactfunc_batch(z)
    
    # High-precision constants
    EPS = dtype.type(1e-15) if dtype == np.float64 else dtype.type(1e-7)
    TINY = dtype.type(1e-16) if dtype == np.float64 else dtype.type(1e-8)
    
    # Inverse sigmoid function (logit)
    f2 = np.clip(-res_batch + er_p + f1, EPS, dtype.type(1.0) - EPS)
    f2i = np.log(f2) - np.log(dtype.type(1.0) - f2)
    
    # Compute the analytical gradient step for reconstruction
    scalar = np.matmul(np.transpose(x_batch, (0, 2, 1)), x_batch)
    denom = (lr * scalar) - lr
    
    # Safe division using dynamic tiny
    denom = np.where(np.abs(denom) < EPS, EPS * np.sign(denom + TINY), denom)

    # Compute the reconstructed ground truth
    er1 = (f2i - z) / denom
    rec = er1 + er_p + f1
    
    return np.clip(rec, dtype.type(0.0), dtype.type(1.0))

def setup_context_mapping(block_size, FIXED_INPUT_SIZE, dtype):
    #This function precomputes the transformation matrix M to average the L context via matrix multiplication
    #Input arguments:
    #block_size: size of processing blocks
    #FIXED_INPUT_SIZE: target size for the output context vector
    #dtype: data type for the matrices

    #Output:
    #M: transformation matrix to apply mean downsampling via matrix multiplication
    #valid_groups: boolean mask indicating valid context vector positions
    
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
            
    return M, valid_groups

# ================================================================================
# 3. WAVEFRONT CPU BATCH PIPELINE
# ================================================================================

def compress_wf_cpu(img, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16):
    #This function compresses an image using a CPU wavefront (diagonal) batch processing approach
    #Input arguments:
    #img: original image to compress
    #block_size: size of processing blocks
    #learning_rate: learning rate for prediction
    #FIXED_INPUT_SIZE: size of context feature vector

    #Output:
    #error_map: compressed representation containing the residuals
    
    img_padded = apply_edge_padding(img, block_size)
    dtype = img_padded.dtype
    h, w = img_padded.shape
    b = block_size
    
    error_map = np.zeros_like(img_padded, dtype=dtype)
    # 1px border buffer for neighbors (Left, Top-Left, Top)
    padded_rec = np.full((h + 1, w + 1), dtype.type(0.5), dtype=dtype)
    
    rows_b, cols_b = h // b, w // b
    M, valid_groups = setup_context_mapping(b, FIXED_INPUT_SIZE, dtype)
    
    for k in range(rows_b + cols_b - 1):
        # 1. Identify blocks belonging to diagonal 'k'
        start_r = max(0, k - cols_b + 1)
        end_r = min(rows_b - 1, k)
        rs = np.arange(start_r, end_r + 1)
        cs = k - rs
        N = len(rs)
        
        py, px = rs * b, cs * b
        
        # 2. Massive extraction (Batch) of Ground Truth
        r_grid = py[:, None, None] + np.arange(b)[None, :, None]
        c_grid = px[:, None, None] + np.arange(b)[None, None, :]
        gt_batch = img_padded[r_grid, c_grid].reshape(N, b * b, 1)
        
        # 3. Massive extraction of Context L
        left_p = padded_rec[py[:, None] + 1 + np.arange(b), px[:, None]]
        tl_p = padded_rec[py[:, None], px[:, None]]
        top_p = padded_rec[py[:, None], px[:, None] + 1 + np.arange(b)]
        
        all_p = np.concatenate([left_p, tl_p, top_p], axis=1)
        
        # 4. Matrix-based Downsampling (Vectorized)
        x_batch = np.matmul(all_p, M) 
        x_batch[:, ~valid_groups] = dtype.type(0.5) #Padding value
        x_batch = x_batch.reshape(N, FIXED_INPUT_SIZE, 1)
        
        # 5. Calculation and Reconstruction
        pred_batch = R2Net_predict_batch(gt_batch, x_batch, learning_rate)
        err_batch = gt_batch - pred_batch
        rec_batch = R2Net_reconstruct_batch(x_batch, err_batch, learning_rate)
        
        # 6. Dump results into the error map and reconstruction buffer
        error_map[r_grid, c_grid] = err_batch.reshape(N, b, b)
        padded_rec[r_grid + 1, c_grid + 1] = rec_batch.reshape(N, b, b)

    return error_map

def decompress_wf_cpu(error_map, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16, original_shape=None):
    #This function decompresses an image using a CPU wavefront (diagonal) batch processing approach
    #Input arguments:
    #error_map: compressed representation containing residuals
    #block_size: size of processing blocks
    #learning_rate: learning rate for reconstruction
    #FIXED_INPUT_SIZE: size of context feature vector
    #original_shape: shape of the image prior to padding

    #Output:
    #reconstructed_img: cropped reconstructed image
    
    dtype = error_map.dtype
    h, w = error_map.shape
    b = block_size
    
    # 1px border buffer for neighbors
    padded_rec = np.full((h + 1, w + 1), dtype.type(0.5), dtype=dtype)
    rows_b, cols_b = h // b, w // b
    M, valid_groups = setup_context_mapping(b, FIXED_INPUT_SIZE, dtype)
    
    for k in range(rows_b + cols_b - 1):
        # 1. Identify blocks belonging to diagonal 'k'
        start_r = max(0, k - cols_b + 1)
        end_r = min(rows_b - 1, k)
        rs = np.arange(start_r, end_r + 1)
        cs = k - rs
        N = len(rs)
        
        py, px = rs * b, cs * b
        r_grid = py[:, None, None] + np.arange(b)[None, :, None]
        c_grid = px[:, None, None] + np.arange(b)[None, None, :]
        
        res_batch = error_map[r_grid, c_grid].reshape(N, b * b, 1)
        
        # 2. Extract Context L from previously reconstructed blocks
        left_p = padded_rec[py[:, None] + 1 + np.arange(b), px[:, None]]
        tl_p = padded_rec[py[:, None], px[:, None]]
        top_p = padded_rec[py[:, None], px[:, None] + 1 + np.arange(b)]
        all_p = np.concatenate([left_p, tl_p, top_p], axis=1)
        
        # 3. Matrix-based Downsampling (Vectorized)
        x_batch = np.matmul(all_p, M).reshape(N, FIXED_INPUT_SIZE, 1)
        x_batch[:, ~valid_groups] = dtype.type(0.5) #Padding value
        
        # 4. Reconstruction and update buffer
        rec_batch = R2Net_reconstruct_batch(x_batch, res_batch, learning_rate)
        padded_rec[r_grid + 1, c_grid + 1] = rec_batch.reshape(N, b, b)
        
    # Remove border buffer and crop to original size
    return crop_to_original(padded_rec[1:, 1:], original_shape)