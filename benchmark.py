import os
import sys
import time
import numpy as np
import pandas as pd
import itertools

# Añadir la carpeta src al path para que Python encuentre tus módulos
sys.path.append(os.path.abspath('src'))

from utils.io import load_raw_image
from utils.normalize import normalize_min_max_values, denormalize_min_max_values
from utils.metrics import calculate_metrics

# =====================================================================
# 1. IMPORTAR TODAS LAS VERSIONES (Asegúrate de que los archivos existan)
# =====================================================================
from coder_original import compress_v1, decompress_v1
from coder_wavefront_cpu import compress_wf_cpu, decompress_wf_cpu
from coder_wavefront_cuda import compress_wf_cuda, decompress_wf_cuda
from coder_parallel_cpu import compress_parallel_cpu, decompress_parallel_cpu
from coder_parallel_cuda import compress_parallel_cuda, decompress_parallel_cuda
from utils.monitor import HardwareMonitor 

# =====================================================================
# 2. MOTOR DE BENCHMARK OPTIMIZADO
# =====================================================================
def run_benchmark(images_dict, modes_dict, baseline_name, block_sizes, learning_rates, fixed_input_sizes, num_runs=3, warmup_runs=1):
    results = []
    # Producto cartesiano incluyendo el tamaño de entrada fijo manual
    combinations = list(itertools.product(images_dict.items(), modes_dict.items(), block_sizes, learning_rates, fixed_input_sizes))
    total_iters = len(combinations)
    
    for idx, ((img_name, img_data), (mode_name, (comp_fn, decomp_fn)), bs, lr, fis) in enumerate(combinations):
        
        print(f"[{idx+1}/{total_iters}] Evaluando: {img_name} | Modo: {mode_name} | BS: {bs} | LR: {lr} | N_Entrada: {fis}")
        
        img = img_data['normalized_array']
        max_val = img_data['max_val'] 
        min_val = img_data['min_val']
        bits = img_data.get('bits', 8)  # Por defecto 8 bits si no se especifica
        
        # Calentamiento (Esencial para inicializar CuPy/Cuda y evitar sesgo de compilación JIT)
        for _ in range(warmup_runs):
            _err = comp_fn(img[:bs*2, :bs*2], block_size=bs, learning_rate=lr, FIXED_INPUT_SIZE=fis)
            _ = decomp_fn(_err, block_size=bs, learning_rate=lr, FIXED_INPUT_SIZE=fis, original_shape=img[:bs*2, :bs*2].shape)
            
        comp_times, decomp_times = [], []
        
        # Pruebas de rendimiento reales
        for run in range(num_runs):
            # Tiempo de Compresión
            start_c = time.perf_counter()
            error_map = comp_fn(img, block_size=bs, learning_rate=lr, FIXED_INPUT_SIZE=fis)
            comp_times.append(time.perf_counter() - start_c)
            
            # Tiempo de Descompresión
            start_d = time.perf_counter()
            reconstructed = decomp_fn(error_map, block_size=bs, learning_rate=lr, FIXED_INPUT_SIZE=fis, original_shape=img.shape)
            decomp_times.append(time.perf_counter() - start_d)
            
        # Cálculo de métricas sobre la imagen denormalizada
        img_original = denormalize_min_max_values(img, min_val, max_val)
        img_reconstructed = denormalize_min_max_values(reconstructed, min_val, max_val)
        metrics_dict = calculate_metrics(img_original, img_reconstructed, bits)
        
        # Cálculo de Throughput (Megapíxeles por segundo)
        num_pixels = img.shape[0] * img.shape[1]
        tiempo_total_medio = np.mean(comp_times) + np.mean(decomp_times)
        throughput_mps = (num_pixels / tiempo_total_medio) / 1_000_000

        results.append({
            'Imagen': img_name,
            'Resolucion': f"{img.shape[1]}x{img.shape[0]}",
            'Pixeles': num_pixels,
            'Modo': mode_name,
            'Block_Size': bs,
            'Fixed_Input_Size': fis,
            'Learning_Rate': lr,
            'Comp_Mean_s': np.mean(comp_times),
            'Decomp_Mean_s': np.mean(decomp_times),
            'Tiempo_Total_s': tiempo_total_medio,
            'Throughput_MP/s': throughput_mps,
            'PAE': metrics_dict['PAE'],
            'MSE': metrics_dict['MSE'],
            'PSNR': metrics_dict['PSNR']
        })
        
    df = pd.DataFrame(results)
    
    # Cálculos de Speedup respecto al baseline (Secuencial)
    print("\nProcesando resultados y calculando Speedup...")
    df_baseline = df[df['Modo'] == baseline_name][['Imagen', 'Block_Size', 'Learning_Rate', 'Fixed_Input_Size', 'Tiempo_Total_s']]
    df_baseline = df_baseline.rename(columns={'Tiempo_Total_s': 'Tiempo_Base_s'})
    df = pd.merge(df, df_baseline, on=['Imagen', 'Block_Size', 'Learning_Rate', 'Fixed_Input_Size'], how='left')
    
    df['Speedup_Total'] = df['Tiempo_Base_s'] / df['Tiempo_Total_s']
    df = df.drop(columns=['Tiempo_Base_s'])
    
    return df

# =====================================================================
# 3. CONFIGURACIÓN Y LANZAMIENTO
# =====================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    PROFILING_MODE = True 
    DTYPE = np.float32  # Recomendado para bloques > 64 y estabilidad
    # ---------------------------------------------------------
    
    print("==================================================")
    print(" INICIANDO R2NET BENCHMARK SUITE (MODO MANUAL)")
    print("==================================================")
    print("Cargando imágenes en memoria...\n")

    # --- CLASE BASE Y MÉDICA ---
    img1_raw = load_raw_image('data/03508649.ube16_1_512_512.raw', 512, 512, 1, 'ube16')
    min1, max1, norm1 = normalize_min_max_values(img1_raw['array'], DTYPE)

    img2_raw = load_raw_image('data/n1_GRAY.ube8_1_2560_2048.raw', 2048, 2560, 1, 'ube8')
    min2, max2, norm2 = normalize_min_max_values(img2_raw['array'], DTYPE)

    img3_raw = load_raw_image('data/kodim01_LUMA_Y.ube8_1_768_512.raw', 512, 768, 1, 'ube8')
    min3, max3, norm3 = normalize_min_max_values(img3_raw['array'], DTYPE)

    # --- CLASE A (2560x1600) ---
    img_cars_raw = load_raw_image('data/A1_2560x1600.raw', 2560, 1600, 1, 'ube8')
    min_cars, max_cars, norm_cars = normalize_min_max_values(img_cars_raw['array'], DTYPE)

    img_crowd_raw = load_raw_image('data/A2_2560x1600.raw', 2560, 1600, 1, 'ube8')
    min_crowd, max_crowd, norm_crowd = normalize_min_max_values(img_crowd_raw['array'], DTYPE)

    img_market_raw = load_raw_image('data/A3_2560x1600.raw', 2560, 1600, 1, 'ube8')
    min_market, max_market, norm_market = normalize_min_max_values(img_market_raw['array'], DTYPE)

    # --- CLASE B (1920x1080) ---
    img_building_raw = load_raw_image('data/B1_1920x1080.raw', 1920, 1080, 1, 'ube8')
    min_building, max_building, norm_building = normalize_min_max_values(img_building_raw['array'], DTYPE)

    img_bucharest_raw = load_raw_image('data/B2_1920x1080.raw', 1920, 1080, 1, 'ube8')
    min_bucharest, max_bucharest, norm_bucharest = normalize_min_max_values(img_bucharest_raw['array'], DTYPE)

    img_library_raw = load_raw_image('data/B3_1920x1080.raw', 1920, 1080, 1, 'ube8')
    min_library, max_library, norm_library = normalize_min_max_values(img_library_raw['array'], DTYPE)

    # --- CLASE C (832x480) ---
    img_rugby_raw = load_raw_image('data/C1_832x480.raw', 832, 480, 1, 'ube8')
    min_rugby, max_rugby, norm_rugby = normalize_min_max_values(img_rugby_raw['array'], DTYPE)

    img_reindeers_raw = load_raw_image('data/C2_832x480.raw', 832, 480, 1, 'ube8')
    min_reindeers, max_reindeers, norm_reindeers = normalize_min_max_values(img_reindeers_raw['array'], DTYPE)

    img_mount_cook_raw = load_raw_image('data/C3_832x480.raw', 832, 480, 1, 'ube8')
    min_mount_cook, max_mount_cook, norm_mount_cook = normalize_min_max_values(img_mount_cook_raw['array'], DTYPE)

    # --- CLASE D (416x240) ---
    img_tomato_raw = load_raw_image('data/D1_416x240.raw', 416, 240, 1, 'ube8')
    min_tomato, max_tomato, norm_tomato = normalize_min_max_values(img_tomato_raw['array'], DTYPE)

    img_flower_raw = load_raw_image('data/D2_416x240.raw', 416, 240, 1, 'ube8')
    min_flower, max_flower, norm_flower = normalize_min_max_values(img_flower_raw['array'], DTYPE)

    img_texture_raw = load_raw_image('data/D3_416x240.raw', 416, 240, 1, 'ube8')
    min_texture, max_texture, norm_texture = normalize_min_max_values(img_texture_raw['array'], DTYPE)

    # --- CLASE E (1280x720) ---
    img_deniro_raw = load_raw_image('data/E1_1280x720.raw', 1280, 720, 1, 'ube8')
    min_deniro, max_deniro, norm_deniro = normalize_min_max_values(img_deniro_raw['array'], DTYPE)

    img_forest_raw = load_raw_image('data/E2_1280x720.raw', 1280, 720, 1, 'ube8')
    min_forest, max_forest, norm_forest = normalize_min_max_values(img_forest_raw['array'], DTYPE)

    img_sea_raw = load_raw_image('data/E3_1280x720.raw', 1280, 720, 1, 'ube8')
    min_sea, max_sea, norm_sea = normalize_min_max_values(img_sea_raw['array'], DTYPE)

    # Diccionario con todas las imágenes activadas
    mis_imagenes = {
        "Base_Medica_512x512":           {"normalized_array": norm1, "max_val": max1, "min_val": min1, "bits": 16},
        "Base_Grande_n1_2560x2048":      {"normalized_array": norm2, "max_val": max2, "min_val": min2, "bits": 8},
        "Base_Kodak01_768x512":          {"normalized_array": norm3, "max_val": max3, "min_val": min3, "bits": 8},
        "A1_Cars_Traffic_2560x1600":     {"normalized_array": norm_cars, "max_val": max_cars, "min_val": min_cars, "bits": 8},
        "A2_Crowd_Enthralled_2560x1600": {"normalized_array": norm_crowd, "max_val": max_crowd, "min_val": min_crowd, "bits": 8},
        "A3_Guelmim_Market_2560x1600":   {"normalized_array": norm_market, "max_val": max_market, "min_val": min_market, "bits": 8},
        "B1_Building_Facade_1920x1080":  {"normalized_array": norm_building, "max_val": max_building, "min_val": min_building, "bits": 8},
        "B2_Bucharest_Univ_1920x1080":   {"normalized_array": norm_bucharest, "max_val": max_bucharest, "min_val": min_bucharest, "bits": 8},
        "B3_Laukaa_Library_1920x1080":   {"normalized_array": norm_library, "max_val": max_library, "min_val": min_library, "bits": 8},
        "C1_Gloucester_Rugby_832x480":   {"normalized_array": norm_rugby, "max_val": max_rugby, "min_val": min_rugby, "bits": 8},
        "C2_Reindeers_Iceland_832x480":  {"normalized_array": norm_reindeers, "max_val": max_reindeers, "min_val": min_reindeers, "bits": 8},
        "C3_Mount_Cook_Park_832x480":    {"normalized_array": norm_mount_cook, "max_val": max_mount_cook, "min_val": min_mount_cook, "bits": 8},
        "D1_Tomato_Rice_416x240":        {"normalized_array": norm_tomato, "max_val": max_tomato, "min_val": min_tomato, "bits": 8},
        "D2_Blue_Star_Flower_416x240":   {"normalized_array": norm_flower, "max_val": max_flower, "min_val": min_flower, "bits": 8},
        "D3_Knit_Texture_416x240":       {"normalized_array": norm_texture, "max_val": max_texture, "min_val": min_texture, "bits": 8},
        "E1_Robert_De_Niro_1280x720":    {"normalized_array": norm_deniro, "max_val": max_deniro, "min_val": min_deniro, "bits": 8},
        "E2_Parkhurst_Forest_1280x720":  {"normalized_array": norm_forest, "max_val": max_forest, "min_val": min_forest, "bits": 8},
        "E3_Sea_Landscape_1280x720":     {"normalized_array": norm_sea, "max_val": max_sea, "min_val": min_sea, "bits": 8},
    }
    
    mis_modos = {
        "1_Secuencial_Original": (compress_v1, decompress_v1),
        "2_Wavefront_CPU": (compress_wf_cpu, decompress_wf_cpu),
        "3_Parallel_CPU": (compress_parallel_cpu, decompress_parallel_cpu),
        "4_Wavefront_CUDA": (compress_wf_cuda, decompress_wf_cuda),
        "5_Parallel_CUDA": (compress_parallel_cuda, decompress_parallel_cuda)
    }
    
    if PROFILING_MODE:
        monitor = HardwareMonitor(interval=0.1)
        monitor.start()

    try:
        # Lanzamiento con tamaños de bloque hasta 256 y FIXED_INPUT_SIZE manual
        df_resultados = run_benchmark(
            images_dict=mis_imagenes,
            modes_dict=mis_modos,
            baseline_name="1_Secuencial_Original", 
            block_sizes=[4, 8, 16, 32, 64, 128, 256], 
            learning_rates=[0.05, 0.1, 0.5, 0.7], 
            fixed_input_sizes=[7, 15, 20, 31], # Define aquí tus valores deseados
            num_runs=5,   
            warmup_runs=1 
        )
    finally:
        if PROFILING_MODE:
            df_hardware = monitor.stop()

    archivo_benchmark = "benchmark_complete_float32.csv"
    df_resultados.to_csv(archivo_benchmark, index=False)
    
    print(f"\n¡Benchmark completado con éxito!")
    print(f"Resultados exportados a: {archivo_benchmark}")
    
    if PROFILING_MODE:
        df_hardware.to_csv("hardware_complete_float32.csv", index=False)