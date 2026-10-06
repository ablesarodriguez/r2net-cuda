import psutil
import threading
import time
import pandas as pd
import numpy as np

# Intentar importar pynvml, manejar el caso si no está instalado o no hay GPU
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Advertencia: pynvml no está instalado. El uso de GPU no será monitorizado. Instálalo con: pip install nvidia-ml-py3")

class HardwareMonitor:
    def __init__(self, interval=0.1):
        """
        interval: Segundos entre cada medición (0.1 = 10 mediciones por segundo)
        """
        self.interval = interval
        self.keep_measuring = False
        self.has_gpu = False
        
        # Inicializar NVIDIA Management Library si está disponible
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Asume la GPU 0
                self.has_gpu = True
            except pynvml.NVMLError:
                print("Advertencia: No se detectó GPU NVIDIA o hubo un error al inicializar pynvml.")

    def _measure_loop(self):
        while self.keep_measuring:
            current_time = time.perf_counter() - self.start_time
            self.timestamps.append(current_time)
            
            # Medir CPU y RAM sin bloquear (interval=None)
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.ram_usage.append(psutil.virtual_memory().used / (1024**3)) # En GB
            
            # Medir GPU y VRAM
            if self.has_gpu:
                try:
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_usage.append(gpu_info.gpu)
                    self.vram_usage.append(mem_info.used / (1024**3)) # En GB
                except pynvml.NVMLError:
                    self.gpu_usage.append(0)
                    self.vram_usage.append(0)
            else:
                self.gpu_usage.append(0)
                self.vram_usage.append(0)
                
            time.sleep(self.interval)

    def start(self):
        # Reiniciar listas de almacenamiento
        self.timestamps = []
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        
        self.keep_measuring = True
        self.start_time = time.perf_counter()
        
        # Iniciar hilo en segundo plano
        self.thread = threading.Thread(target=self._measure_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        
        # Apagar pynvml si fue inicializado
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
                
        # Asegurarse de que todas las listas tengan la misma longitud antes de crear el DataFrame
        min_len = min(len(self.timestamps), len(self.cpu_usage), len(self.ram_usage), len(self.gpu_usage), len(self.vram_usage))
        
        return pd.DataFrame({
            'Tiempo_s': self.timestamps[:min_len],
            'CPU_%': self.cpu_usage[:min_len],
            'RAM_GB': self.ram_usage[:min_len],
            'GPU_%': self.gpu_usage[:min_len],
            'VRAM_GB': self.vram_usage[:min_len]
        })