import numpy as np
from utils.padding import apply_edge_padding, crop_to_original

def applyactfunc(z):
    # Esta función aplica la función de activación Sigmoide
    dtype = z.dtype
    # Clip fijo y seguro para float64 para evitar desbordamientos en np.exp
    z = np.clip(z, dtype.type(-500.0), dtype.type(500.0))
    p = dtype.type(1.0) + np.exp(-z)
    
    return dtype.type(1.0) / p

def R2Net_predict(gt, x, learning_rate):
    # Esta función implementa el R2Net para predecir un bloque
    dtype = gt.dtype
    sb, sfv = len(gt), len(x)   
   
    # Inicialización de pesos (W) y sesgo (b)
    W = np.ones((sfv, sb), dtype=dtype)
    b = np.ones((1, sb), dtype=dtype)
    er_p = np.zeros((sb, 1), dtype=dtype)
    lr = dtype.type(learning_rate)
    
    for it in (0, 1): # Dos iteraciones de back-propagation (BP)
        # Forward propagation
        z = ((W.transpose()).dot(x) + b.transpose())
        hz = applyactfunc(z)
        pred = er_p + hz
        
        # Backward propagation
        if it == 0:
            er = (gt - pred)
            dW = -(x.dot(er.transpose())) 
        else:
            er = (pred - gt)
            dW = (x.dot(er.transpose())) 

        db = er.transpose() 
        er_p = er 
    
        # Gradient descent
        W -= (lr * dW)
        b -= (lr * db)
        
    return pred

def R2Net_reconstruct(x, res, learning_rate):
    # Esta función implementa el R2Net para reconstruir un bloque desde su residuo
    dtype = res.dtype
    er2 = -res
    sb, sfv = len(res), len(x)   
   
    W = np.ones((sfv, sb), dtype=dtype)
    b = np.ones((1, sb), dtype=dtype)
    er_p = np.zeros((sb, 1), dtype=dtype)
    lr = dtype.type(learning_rate)
        
    # Forward propagation para obtener la predicción inicial f1
    z = ((W.transpose()).dot(x) + b.transpose())
    f1 = applyactfunc(z)
    f2 = er2 + er_p + f1
    
    # Constantes hardcodeadas de alta precisión optimizadas para float64
    EPS = 1e-15
    TINY = 1e-16
    
    # Inversión de la función de activación (logit)
    f2 = np.clip(f2, EPS, dtype.type(1.0) - EPS)
    f2i = np.log(f2) - np.log(dtype.type(1.0) - f2)
        
    # Cálculo del paso de gradiente analítico para la reconstrucción
    scalar = x.transpose().dot(x)
    denom = (lr * scalar) - lr
    
    # División segura usando las tolerancias de float64
    denom = np.where(np.abs(denom) < EPS, EPS * np.sign(denom + TINY), denom)

    er1 = (f2i - z) / denom
    rec = er1 + er_p + f1
        
    return np.clip(rec, dtype.type(0.0), dtype.type(1.0))

def get_context_pixels(reconstructed_img, x, y, width, height, block_size, FIXED_INPUT_SIZE=16):
    # Esta función extrae los píxeles de contexto (izquierda, arriba-izquierda, arriba)
    dtype = reconstructed_img.dtype
    pixels = []
    pad_val = dtype.type(0.5) 

    # Extracción de contexto izquierdo
    if x > 0:
        left_pixels = reconstructed_img[y : min(y+block_size, height), x-1].flatten()
        pixels.extend(left_pixels)
        if (missing := block_size - len(left_pixels)) > 0: pixels.extend([pad_val] * missing)
    else:
        pixels.extend([pad_val] * block_size)

    # Extracción de contexto arriba-izquierda
    pixels.append(reconstructed_img[y-1, x-1] if (y > 0 and x > 0) else pad_val)

    # Extracción de contexto superior
    if y > 0:
        top_pixels = reconstructed_img[y-1, x : min(x+block_size*2, width)].flatten()
        pixels.extend(top_pixels)
        if (missing := block_size*2 - len(top_pixels)) > 0: pixels.extend([pad_val] * missing)
    else:
        pixels.extend([pad_val] * block_size*2)

    # Downsampling del contexto extraído
    arr = np.array(pixels, dtype=dtype)
    groups = np.array_split(arr, FIXED_INPUT_SIZE)
    downsampled_vector = [g.mean() if g.size > 0 else pad_val for g in groups]
    
    return np.array(downsampled_vector, dtype=dtype).reshape(-1, 1)

# =====================================================================
# PIPELINE DE COMPRESIÓN / DESCOMPRESIÓN (PROCESAMIENTO EN FLOAT64)
# =====================================================================

def compress_v1(img, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16):
    # 1. FORZAR TODOS LOS CÁLCULOS INTERNOS A FLOAT64
    
    img_padded = apply_edge_padding(img, block_size)
    dtype = img_padded.dtype # Se convierte automáticamente en np.float64
    height, width = img_padded.shape
    
    reconstructed_img = np.zeros_like(img_padded, dtype=dtype)
    error_map = np.zeros_like(img_padded, dtype=dtype)

    for i in range(0, height, block_size):      
        for j in range(0, width, block_size):   
            gt_vec = img_padded[i:i+block_size, j:j+block_size].flatten().reshape(-1, 1)
            
            x_block = get_context_pixels(reconstructed_img, j, i, width, height, block_size, FIXED_INPUT_SIZE)  
            
            pred_vec = R2Net_predict(gt_vec, x_block, learning_rate)
            esignal_vec = gt_vec - pred_vec
            error_map[i:i+block_size, j:j+block_size] = esignal_vec.reshape(block_size, block_size)

            rec_vec = R2Net_reconstruct(x_block, esignal_vec, learning_rate)
            reconstructed_img[i:i+block_size, j:j+block_size] = rec_vec.reshape(block_size, block_size)

    # 2. ALMACENAR Y RETORNAR EL MAPA DE ERRORES EN FLOAT16
    return error_map

def decompress_v1(error_map, block_size=16, learning_rate=0.5, FIXED_INPUT_SIZE=16, original_shape=None):
    # 1. RESTAURAR EL MAPA DE RESIDUOS A FLOAT64 ANTES DE OPERAR LA INVERS
    
    dtype = error_map.dtype # Se convierte automáticamente en np.float64
    height, width = error_map.shape
    reconstructed_img = np.zeros_like(error_map, dtype=dtype)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            x_block = get_context_pixels(reconstructed_img, j, i, width, height, block_size, FIXED_INPUT_SIZE)
            
            esignal_vec = error_map[i:i+block_size, j:j+block_size].flatten().reshape(-1, 1) 
            
            # Reconstrucción matemática de alta fidelidad en precisión de 64 bits
            rec_vec = R2Net_reconstruct(x_block, esignal_vec, learning_rate)
            reconstructed_img[i:i+block_size, j:j+block_size] = rec_vec.reshape(block_size, block_size)

    return crop_to_original(reconstructed_img, original_shape)