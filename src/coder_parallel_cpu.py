import numpy as np
from utils.padding import apply_edge_padding, crop_to_original

def applyactfunc_batch(z):
    #This function applies the Sigmoid activation function to a batch of inputs with a safety limit
    #Input arguments:
    #z: input batch tensor for activation
    
    #Output:
    #result: activated batch values using the Sigmoid function bounded to [0,1]
    
    dtype = z.dtype
    #Fixed and safe clip to prevent overflows in np.exp
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
    
    #Initialize weights (W) and bias (b) to one
    W = np.ones((batch_size, sfv, sb), dtype=dtype)
    b = np.ones((batch_size, 1, sb), dtype=dtype)
    er_p = np.zeros((batch_size, sb, 1), dtype=dtype)
    lr = dtype.type(learning_rate)

    for it in (0, 1): #Two iterations of back-propagation (BP)
        #Forward propagation
        z = np.matmul(np.transpose(W, (0, 2, 1)), x_batch) + np.transpose(b, (0, 2, 1))
        hz = applyactfunc_batch(z)
        pred = er_p + hz
        
        #Backward propagation
        if it == 0:
            er = (gt_batch - pred)
            dW = -(np.matmul(x_batch, np.transpose(er, (0, 2, 1))))
        else:
            er = (pred - gt_batch)
            dW = np.matmul(x_batch, np.transpose(er, (0, 2, 1)))
        
        #Gradient descent (weight update)
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
    batch_size = res_batch.shape[0]
    sb = res_batch.shape[1]  # sb = size of res
    sfv = x_batch.shape[1]   # sfv = size of x
    lr = dtype.type(learning_rate)
    
    #Initialize weights (W) and bias (b) to one
    W = np.ones((batch_size, sfv, sb), dtype=dtype)
    b = np.ones((batch_size, 1, sb), dtype=dtype)
    er_p = np.zeros((batch_size, sb, 1), dtype=dtype)
    
    #Forward propagation to obtain the initial prediction f1
    z = np.matmul(np.transpose(W, (0, 2, 1)), x_batch) + np.transpose(b, (0, 2, 1))
    f1 = applyactfunc_batch(z)
    
    #High-precision constants
    EPS = dtype.type(1e-15) if dtype == np.float64 else dtype.type(1e-7)
    
    #Inverse sigmoid function (logit)
    f2 = np.clip(-res_batch + er_p + f1, EPS, dtype.type(1.0) - EPS)
    f2i = np.log(f2) - np.log(dtype.type(1.0) - f2)
    
    #Compute the analytical gradient step for reconstruction
    scalar = np.matmul(np.transpose(x_batch, (0, 2, 1)), x_batch)
    denom = (lr * scalar) - lr
    
    #Safe division
    denom = np.where(np.abs(denom) < EPS, EPS * np.sign(denom + dtype.type(1e-16)), denom)

    #Compute the reconstructed ground truth
    er1 = (f2i - z) / denom
    rec = er1 + er_p + f1
    
    return np.clip(rec, dtype.type(0.0), dtype.type(1.0))


def setup_context_mapping(block_size, FIXED_INPUT_SIZE, dtype):
    #This function creates a mapping matrix to downsample context pixels to a fixed size
    #Input arguments:
    #block_size: size of the processing block
    #FIXED_INPUT_SIZE: target size for the downsampled context vector
    #dtype: data type of the tensors
    
    #Output:
    #M: mapping matrix for downsampling, organized as a 2D array
    #valid_groups: boolean array indicating which groups contain valid pixels
    
    L = (2 * block_size) - 1
    K = FIXED_INPUT_SIZE
    splits = np.array_split(np.zeros(L), K)
    
    M = np.zeros((L, K), dtype=dtype)
    valid_groups = np.zeros(K, dtype=bool)
    
    idx = 0
    for i, g in enumerate(splits):
        s = len(g)
        if s > 0:
            M[idx:idx+s, i] = dtype.type(1.0) / dtype.type(s)
            idx += s
            valid_groups[i] = True
            
    return M, valid_groups

# Ensure R2Net_predict_batch and R2Net_reconstruct_batch 
# use the DTYPE that arrives in the tensors.

def compress_parallel_cpu(img, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16):
    #This function compresses an image in parallel by batch processing its blocks
    #Input arguments:
    #img: original input image
    #block_size: processing block size
    #learning_rate: learning rate for R2Net
    #FIXED_INPUT_SIZE: context vector size
    
    #Output:
    #error_map_final: residual or error map resulting from the parallel compression
    #context_raw: tuple containing raw top and left context pixels
    
    DTYPE = img.dtype
    img_padded = apply_edge_padding(img, block_size)
    h, w = img_padded.shape
    b = block_size
    
    # 1. Block extraction
    blocks = img_padded.reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b, b)
    N = blocks.shape[0]
    
    # 2. Raw context extraction (unquantized)
    ctx_top = blocks[:, 0, :].copy()
    ctx_left = blocks[:, :, 0].copy()
    
    # 3. Context mapping and downsampling
    all_pixels = np.concatenate([ctx_left, ctx_top[:, 1:]], axis=1)
    M, valid_groups = setup_context_mapping(b, FIXED_INPUT_SIZE, DTYPE)
    
    x_batch = np.matmul(all_pixels, M)
    x_batch[:, ~valid_groups] = DTYPE.type(0.5)
    x_batch = x_batch.reshape(N, FIXED_INPUT_SIZE, 1)
    
    # 4. Prediction and direct error calculation (pure float)
    gt_batch = blocks.reshape(N, b * b, 1).astype(DTYPE)
    pred_batch = R2Net_predict_batch(gt_batch, x_batch, DTYPE.type(learning_rate))
    
    error_map = (gt_batch - pred_batch).astype(DTYPE)
    
    # Reintegrate context into the error map to maintain structure
    err_blocks = error_map.reshape(N, b, b)
    err_blocks[:, 0, :] = ctx_top
    err_blocks[:, :, 0] = ctx_left
    
    error_map_final = err_blocks.reshape(h // b, w // b, b, b).transpose(0, 2, 1, 3).reshape(h, w)
    
    return error_map_final, (ctx_top, ctx_left)

def decompress_parallel_cpu(error_map, context_raw, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16, original_shape=None):
    #This function decompresses an image in parallel from its error map and raw context
    #Input arguments:
    #error_map: stored residual map
    #context_raw: tuple containing raw top and left context pixels
    #block_size: processing block size
    #learning_rate: learning rate for R2Net
    #FIXED_INPUT_SIZE: context vector size
    #original_shape: original image dimensions before padding
    
    #Output:
    #reconstructed_img: final reconstructed image cropped to its original size
    
    rec_top, rec_left = context_raw
    DTYPE = error_map.dtype
    h, w = error_map.shape
    b = block_size
    
    # Block reshaping
    err_blocks = error_map.reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b, b)
    N = err_blocks.shape[0]
    
    # Context mapping and downsampling
    all_pixels = np.concatenate([rec_left, rec_top[:, 1:]], axis=1)
    M, valid_groups = setup_context_mapping(b, FIXED_INPUT_SIZE, DTYPE)
    
    x_batch = np.matmul(all_pixels, M)
    x_batch[:, ~valid_groups] = DTYPE.type(0.5)
    x_batch = x_batch.reshape(N, FIXED_INPUT_SIZE, 1)
    
    # Pure mathematical reconstruction
    res_batch = err_blocks.reshape(N, b * b, 1).astype(DTYPE)
    rec_batch = R2Net_reconstruct_batch(x_batch, res_batch, DTYPE.type(learning_rate))
    
    # Reintegrate context into reconstructed blocks
    rec_blocks = rec_batch.reshape(N, b, b)
    rec_blocks[:, 0, :] = rec_top
    rec_blocks[:, :, 0] = rec_left
    
    reconstructed_img = rec_blocks.reshape(h // b, w // b, b, b).transpose(0, 2, 1, 3).reshape(h, w)
    return crop_to_original(reconstructed_img, original_shape)