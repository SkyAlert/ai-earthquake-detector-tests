#!/usr/bin/env python3
"""
CREIME_RT MONITOR - Sistema de Alerta Sísmica en Tiempo Real
Monitor que usa la lógica del simulador pero con datos en tiempo real de AnyShake
"""

import os
import socket
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime
import logging
from scipy.signal import butter, lfilter
import psutil
import gc
import queue
import json
import sys
import uuid
from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats

# ===== CONFIGURACIÓN GPU SEGURA PARA JETSON ORIN NANO =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configuración de logging optimizada
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "creime_rt_monitor.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# ===== CONFIGURACIÓN DE VISUALIZACIÓN =====
VISUALIZATION_ENABLED = True
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Backend interactivo seguro
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import style
    style.use('fivethirtyeight')
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib no disponible - visualización desactivada")
except Exception as e:
    VISUALIZATION_ENABLED = False
    logging.warning(f"Error configurando matplotlib: {e} - visualización desactivada")

# Constantes de visualización
DISPLAY_SECONDS = 60
SAMPLING_RATE = 100
MAX_VISUALIZATION_SAMPLES = DISPLAY_SECONDS * SAMPLING_RATE

# Constantes de conversión
SENSITIVITY = 0.122
MG_TO_GALS = 0.980665
CONVERSION_FACTOR = SENSITIVITY * MG_TO_GALS

# Colores personalizados
COLOR_TEAL = '#009688'
COLOR_ORANGE = '#FF9800'
COLOR_RED = '#FF0000'
COLOR_GREEN = '#00FF00'

class RealTimeVisualizer:
    """Sistema de visualización para monitor en tiempo real"""
    
    def __init__(self, seismic_detector):
        self.detector = seismic_detector
        self.running = False
        self.fig = None
        self.axes = None
        self.animation = None
        self.visualization_enabled = VISUALIZATION_ENABLED
        
        # Buffers de datos
        self.times = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        self.data_enz = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        self.data_ene = deque(maxlen=MAX_VISUALIZATION_SAMPLES) 
        self.data_enn = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        
        # Líneas de gráfico
        self.line_enz, self.line_ene, self.line_enn = None, None, None
        self.info_text = None
        
        # Marcadores de ventana CREIME_RT
        self.creime_markers = []
        self.processing_window_markers = []
        
        # Historial de máximos para ajuste suave
        self.max_values_history = {
            'ENZ': deque(maxlen=10),
            'ENE': deque(maxlen=10),
            'ENN': deque(maxlen=10)
        }
        
        # Estadísticas
        self.packet_count = 0
        self.start_time = time.time()
        
        self.lock = threading.Lock()
        self.setup_visualization()
    
    def setup_visualization(self):
        """Configuración de visualización para monitor"""
        if not self.visualization_enabled:
            return
            
        try:
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
            
            self.fig, self.axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120)
            self.ax1, self.ax2, self.ax3 = self.axes
            
            self.fig.suptitle('CREIME_RT MONITOR - Datos en Tiempo Real', 
                             fontsize=16, fontweight='bold')
            
            self.line_enz, = self.ax1.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENZ')
            self.line_ene, = self.ax2.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENE')
            self.line_enn, = self.ax3.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENN')
            
            components_config = [
                (self.ax1, 'Componente Vertical (ENZ)', COLOR_TEAL),
                (self.ax2, 'Componente Este-Oeste (ENE)', COLOR_TEAL),
                (self.ax3, 'Componente Norte-Sur (ENN)', COLOR_TEAL)
            ]
            
            for ax, title, color in components_config:
                ax.set_ylabel('Aceleración (Gals)', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.legend(loc='upper right')
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            self.ax3.set_xlabel('Tiempo (segundos)', fontsize=12)
            
            self.info_text = self.ax3.text(0.02, 0.95, '', transform=self.ax3.transAxes, 
                                          fontsize=10, verticalalignment='top', color=COLOR_ORANGE,
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configurar marcadores de ventana de procesamiento
            self.setup_processing_markers()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Configurar evento de cierre de ventana
            self.fig.canvas.mpl_connect('close_event', self.on_window_close)
            
            logging.info("Visualizador del monitor configurado correctamente")
            
        except Exception as e:
            logging.error(f"Error configurando visualización: {e}")
            self.visualization_enabled = False
    
    def setup_processing_markers(self):
        """Configura marcadores de ventana de procesamiento CREIME_RT"""
        if not self.visualization_enabled:
            return
            
        try:
            for ax in [self.ax1, self.ax2, self.ax3]:
                if ax is None:
                    continue
                # Línea vertical para marcar ventana actual de procesamiento
                window_line = ax.axvline(x=0, color=COLOR_GREEN, linestyle='-', 
                                       linewidth=3, alpha=0.8, label='Ventana CREIME_RT (30s)')
                # Área sombreada para ventana de procesamiento
                window_fill = ax.axvspan(0, 30, color=COLOR_GREEN, alpha=0.15)
                
                self.processing_window_markers.append({
                    'line': window_line,
                    'fill': window_fill
                })
            
            # Agregar leyenda solo en el primer subplot
            self.ax1.legend(loc='upper left', fontsize=9)
            
        except Exception as e:
            logging.error(f"Error configurando marcadores de procesamiento: {e}")
    
    def calculate_dynamic_ylimits(self, data, component_name):
        """Escalado dinámico simple basado en amplitud real de la señal"""
        if len(data) == 0:
            return -0.1, 0.1
        
        # Obtener valores máximo y mínimo reales
        data_max = np.max(data)
        data_min = np.min(data)
        
        # Calcular rango con margen del 10%
        data_range = data_max - data_min
        margin = data_range * 0.1 if data_range > 0 else 0.01
        
        # Límites dinámicos basados en amplitud real
        y_max = data_max + margin
        y_min = data_min - margin
        
        # Límite mínimo para evitar escalas muy pequeñas
        if abs(y_max - y_min) < 0.02:
            center = (y_max + y_min) / 2
            y_max = center + 0.01
            y_min = center - 0.01
        
        return y_min, y_max
    
    def update_data(self, component, data, timestamp):
        """Actualización de datos para visualización en tiempo real"""
        if not self.visualization_enabled:
            return
            
        current_time = time.time()
        
        with self.lock:
            self.packet_count += 1
            
            for i, value in enumerate(data):
                sample_time = current_time - (len(data) - i) * (1.0 / SAMPLING_RATE)
                
                if component == 'ENZ':
                    self.times.append(sample_time)
                    self.data_enz.append(value)
                elif component == 'ENE':
                    self.data_ene.append(value)
                elif component == 'ENN':
                    self.data_enn.append(value)
    
    def update_plot(self, frame):
        """Actualización de gráficos"""
        if not self.visualization_enabled or not self.fig:
            return []
        
        with self.lock:
            times_copy = np.array(self.times)
            enz_copy = np.array(self.data_enz)
            ene_copy = np.array(self.data_ene)
            enn_copy = np.array(self.data_enn)
        
        min_len = min(len(times_copy), len(enz_copy), len(ene_copy), len(enn_copy))
        if min_len < 10:
            return []
        
        times_trim = times_copy[-min_len:]
        enz_trim = enz_copy[-min_len:]
        ene_trim = ene_copy[-min_len:]
        enn_trim = enn_copy[-min_len:]
        
        current_time_sec = time.time()
        rel_times = current_time_sec - times_trim
        
        self.line_enz.set_data(rel_times, enz_trim)
        self.line_ene.set_data(rel_times, ene_trim)
        self.line_enn.set_data(rel_times, enn_trim)
        
        xlim = (DISPLAY_SECONDS, 0)
        for ax in self.axes:
            ax.set_xlim(xlim)
        
        ylim_enz = self.calculate_dynamic_ylimits(enz_trim, 'ENZ')
        ylim_ene = self.calculate_dynamic_ylimits(ene_trim, 'ENE')
        ylim_enn = self.calculate_dynamic_ylimits(enn_trim, 'ENN')
        
        self.ax1.set_ylim(ylim_enz)
        self.ax2.set_ylim(ylim_ene)
        self.ax3.set_ylim(ylim_enn)
        
        # Actualizar marcadores de ventana de procesamiento
        self.update_processing_markers(rel_times)
        
        current_time_str = datetime.now().strftime('%H:%M:%S')
        run_time = time.time() - self.start_time
        packets_per_sec = self.packet_count / run_time if run_time > 0 else 0
        
        detector_info = ""
        if hasattr(self.detector, 'detection_count'):
            detector_info = f" | Detecciones: {self.detector.detection_count}"
        if hasattr(self.detector, 'processing_count'):
            detector_info += f" | Ventanas: {self.detector.processing_count}"
        
        info_text = (f"Tiempo: {current_time_str} | "
                    f"Paquetes: {self.packet_count} | "
                    f"Rate: {packets_per_sec:.1f} pkt/s | "
                    f"Muestras: {min_len}{detector_info}")
        
        self.info_text.set_text(info_text)
        
        return []
    
    def update_processing_markers(self, rel_times):
        """Actualiza marcadores de ventana de procesamiento CREIME_RT"""
        if not self.processing_window_markers or len(rel_times) == 0:
            return
            
        try:
            # La ventana de procesamiento son los últimos 30 segundos
            window_start = 30  # 30 segundos atrás
            window_end = 0     # Tiempo actual
            
            for markers in self.processing_window_markers:
                # Actualizar línea de ventana actual
                markers['line'].set_xdata([window_end, window_end])
                
                # Actualizar área sombreada de ventana
                try:
                    markers['fill'].remove()
                    ax = markers['line'].axes
                    markers['fill'] = ax.axvspan(window_start, window_end, 
                                                color=COLOR_GREEN, alpha=0.15)
                except:
                    pass
                    
        except Exception as e:
            logging.debug(f"Error actualizando marcadores: {e}")
    
    def start_visualization(self):
        """Inicia visualización en thread principal"""
        if not self.visualization_enabled or not self.fig:
            return
            
        try:
            self.running = True
            # Crear animación sin blit para evitar problemas de threading
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=200, blit=False, cache_frame_data=False
            )
            logging.info("Visualizador del monitor iniciado")
        except Exception as e:
            logging.error(f"Error iniciando visualización: {e}")
            self.visualization_enabled = False
    
    def on_window_close(self, event):
        """Maneja el cierre de la ventana del visualizador"""
        logging.info("Ventana cerrada - Deteniendo sistema")
        self.detector.stop_monitor()
    
    def stop_visualization(self):
        """Detiene el sistema de visualización"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        logging.info("Visualizador detenido")

class UltraFastBuffer:
    """Buffer de latencia mínima - ventana deslizante continua"""
    
    def __init__(self, window_size=3000, sampling_rate=100, update_interval=1.0):
        self.window_size = int(window_size)
        self.sampling_rate = int(sampling_rate)
        self.update_interval = update_interval  # 1000ms sincronizado con AnyShake
        
        # Buffer circular exacto para ventana de 30s
        self.buffers = {
            'ENZ': deque(maxlen=self.window_size),
            'ENE': deque(maxlen=self.window_size),
            'ENN': deque(maxlen=self.window_size)
        }
        
        self.lock = threading.Lock()
        self.window_count = 0
        self.last_window_time = 0
        self.ready = False
        self.min_ready_samples = self.window_size  # Requiere ventana completa
        self.new_data_event = threading.Event()
        
        self.performance_stats = {
            'windows_generated': 0,
            'data_points_received': 0,
            'last_update': time.time()
        }
    
    def add_data(self, component, data, timestamp):
        """Añade datos del monitor en tiempo real"""
        with self.lock:
            if component in self.buffers:
                self.buffers[component].extend(data)
                self.performance_stats['data_points_received'] += len(data)
                
                min_samples = min(len(buf) for buf in self.buffers.values())
                self.ready = min_samples >= self.min_ready_samples
                
                if data:
                    self.new_data_event.set()
    
    def wait_for_new_data(self, timeout=0.5):
        """Espera por nuevos datos"""
        return self.new_data_event.wait(timeout)
    
    def reset_data_event(self):
        """Resetea el evento de nuevos datos"""
        self.new_data_event.clear()
    
    def get_latest_window(self):
        """Ventana deslizante de latencia mínima - siempre los últimos 30s"""
        with self.lock:
            current_time = time.time()
            
            # Procesar cada 10ms para latencia mínima
            if current_time - self.last_window_time < self.update_interval:
                return None
                
            if not self.ready:
                return None
            
            window_data = []
            for component in ['ENZ', 'ENE', 'ENN']:
                buf = self.buffers[component]
                
                # Usar directamente el buffer circular (siempre los últimos 30s)
                if len(buf) == self.window_size:
                    component_data = list(buf)  # Los últimos 3000 datos
                else:
                    # Rellenar solo si no tenemos ventana completa aún
                    padding_needed = self.window_size - len(buf)
                    component_data = [0.0] * padding_needed + list(buf)
                
                window_data.append(component_data)
            
            try:
                window_3d = np.stack([window_data[0], window_data[1], window_data[2]], axis=1)
                window_3d = np.expand_dims(window_3d, axis=0).astype(np.float32)
                
                self.window_count += 1
                self.performance_stats['windows_generated'] += 1
                self.last_window_time = current_time
                
                return window_3d
                
            except Exception as e:
                logging.error(f"Error creando ventana 3D: {e}")
                return None
    
    def get_buffer_status(self):
        """Estado del buffer"""
        with self.lock:
            status = {}
            for comp, buf in self.buffers.items():
                status[comp] = {
                    'samples': len(buf),
                    'percent': (len(buf) / self.window_size) * 100,
                    'seconds': len(buf) / self.sampling_rate
                }
            
            status['ready'] = self.ready
            status['min_required'] = f"{self.min_ready_samples} muestras ({self.min_ready_samples/self.sampling_rate:.1f}s)"
            
            return status

class OptimizedHybridFilter:
    """Filtro optimizado: Z-Score → Filtro 1-45Hz → Conversión Gals"""
    
    def __init__(self, fs=100, hp_cutoff=1.0, lp_cutoff=45):
        self.fs = fs
        nyquist = 0.5 * fs
        
        normal_cutoff_hp = hp_cutoff / nyquist
        self.b_hp, self.a_hp = butter(2, normal_cutoff_hp, btype='high', analog=False)
        
        normal_cutoff_lp = lp_cutoff / nyquist
        self.b_lp, self.a_lp = butter(4, normal_cutoff_lp, btype='low', analog=False)
        
        self.zi_hp = np.zeros(max(len(self.a_hp), len(self.b_hp)) - 1)
        self.zi_lp = np.zeros(max(len(self.a_lp), len(self.b_lp)) - 1)
        
        self.zscore_buffer = deque(maxlen=fs)  # Buffer para Z-Score
    
    def apply_filter(self, data):
        """Pipeline: Z-Score → Filtro 1-45Hz → Conversión Gals"""
        if not data:
            return data
        
        # 1. Normalización Z-Score (reemplaza detrending)
        self.zscore_buffer.extend(data)
        if len(self.zscore_buffer) > 1:
            buffer_data = list(self.zscore_buffer)
            mean_val = np.mean(buffer_data)
            std_val = np.std(buffer_data)
            if std_val > 0:
                zscore_data = [(x - mean_val) / std_val for x in data]
            else:
                zscore_data = [0.0] * len(data)
        else:
            zscore_data = [0.0] * len(data)
            
        zscore_np = np.array(zscore_data, dtype=np.float32)
        
        # 2. Filtro pasa-altas (1 Hz)
        filtered_hp, self.zi_hp = lfilter(self.b_hp, self.a_hp, zscore_np, zi=self.zi_hp)
        
        # 3. Filtro pasa-bajas (45 Hz)
        filtered_lp, self.zi_lp = lfilter(self.b_lp, self.a_lp, filtered_hp, zi=self.zi_lp)
        
        # 4. Conversión a Gals
        gals_data = filtered_lp * CONVERSION_FACTOR
        
        return gals_data.astype(np.float32).tolist()

class UltraFastProcessingPipeline:
    """Pipeline de procesamiento para monitor"""
    
    def __init__(self, model_path, num_workers=1):
        self.model_path = model_path
        self.num_workers = num_workers
        self.processing_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        
        self.start_workers()
    
    def start_workers(self):
        """Inicia workers de procesamiento"""
        self.running = True
        self.workers_ready = threading.Event()  # Evento para sincronizar inicialización
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self.processing_worker,
                name=f"CREIME_Monitor_Worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Iniciados {self.num_workers} workers del monitor")
    
    def processing_worker(self):
        """Worker para CREIME_RT en monitor"""
        try:
            from saipy.models.creime import CREIME_RT
            model = CREIME_RT(self.model_path)
            logging.info("Worker monitor CREIME_RT inicializado")
            
            # Señalar que el worker está listo
            self.workers_ready.set()
            
        except Exception as e:
            logging.error(f"Worker monitor no pudo cargar modelo: {e}")
            return
        
        while self.running:
            try:
                window_data, processing_id = self.processing_queue.get(timeout=1.0)
                
                if window_data is None:
                    break
                
                start_time = time.time()
                
                try:
                    y_pred, predictions = model.predict(window_data)
                    
                    # Usar el valor máximo del vector completo (6000 muestras)
                    if y_pred is not None and len(y_pred.shape) > 1:
                        raw_output = float(np.max(y_pred[0]))  # Máximo del vector completo
                    else:
                        raw_output = -4.0
                    
                    if raw_output > -0.5:
                        detection = 1
                        magnitude = raw_output if raw_output > 0 else 0.0
                    else:
                        detection = 0
                        magnitude = None
                    
                    result = (detection, magnitude, raw_output)
                    
                except (IndexError, ValueError, TypeError) as e:
                    logging.warning(f"Error en predicción monitor: {e}")
                    result = (0, None, -4.0)
                except Exception as e:
                    logging.error(f"Error inesperado en monitor: {e}")
                    result = (0, None, -4.0)
                
                processing_time = time.time() - start_time
                
                self.result_queue.put({
                    'result': result,
                    'processing_id': processing_id,
                    'processing_time': processing_time,
                    'window_data': window_data
                })
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error en worker monitor: {e}")
                try:
                    self.processing_queue.task_done()
                except:
                    pass
                continue
    
    def submit_window(self, window_data, processing_id):
        """Envía ventana para procesamiento"""
        try:
            self.processing_queue.put((window_data, processing_id), timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.5):
        """Obtiene resultado"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_worker_ready(self, timeout=30):
        """Verifica si el worker CREIME_RT está listo"""
        return hasattr(self, 'workers_ready') and self.workers_ready.wait(timeout)
    
    def stop_workers(self):
        """Detiene workers"""
        self.running = False
        for _ in range(self.num_workers):
            try:
                self.processing_queue.put((None, None), timeout=0.5)
            except queue.Full:
                break
        
        for worker in self.workers:
            worker.join(timeout=2.0)

class RealTimeMonitor:
    """
    Monitor principal que usa datos en tiempo real de AnyShake
    """
    
    def __init__(self, model_path, host='localhost', port=30000, sampling_rate=100):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        
        # Parámetros sincronizados con AnyShake (datos cada 1000ms)
        self.window_size = 30 * sampling_rate  # 3000 muestras - 30 SEGUNDOS
        self.latency_target = 1.0  # 1000ms - perfectamente sincronizado
        self.detection_threshold = -0.5  # Umbral original CREIME_RT
        self.noise_baseline = -4.0
        self.high_noise_threshold = -1.80
        self.magnitude_threshold = 0.0  # Umbral original para magnitud
        self.consecutive_windows = 1  # Confirmación inmediata para latencia mínima
        
        # Componentes del sistema
        self.buffer = UltraFastBuffer(
            window_size=self.window_size,
            sampling_rate=sampling_rate,
            update_interval=self.latency_target  # 10ms para latencia mínima
        )
        
        self.hybrid_filter = OptimizedHybridFilter(fs=sampling_rate)
        self.processing_pipeline = UltraFastProcessingPipeline(model_path, num_workers=1)
        self.visualizer = RealTimeVisualizer(self)
        
        # Estado del sistema
        self.running = False
        self.socket = None
        self.data_buffer = b''
        
        # Estadísticas
        self.detection_count = 0
        self.last_detection_time = None
        self.packet_count = 0
        self.start_time = None
        self.processing_count = 0
        self.last_processing_time = 0
        self.detection_buffer = deque(maxlen=self.consecutive_windows)
        
        # Configuración de estación
        self.station_id = "CREIME_RT_MONITOR"
        
        # Rastreo de eventos y timestamps
        self.detected_events = []
        self.monitor_start_time = None
        
        # Rastreo de tiempos de detección
        self.first_detection_time = None  # Primer evento detectado
        self.first_confirmation_time = None  # Primera confirmación de sismo
        
        # Rastreo de valores CREIME_RT para diagnóstico
        self.creime_values = []
        self.creime_timestamps = []  # Timestamps correspondientes
        
        # Buffer para MiniSEED (15 segundos antes del evento)
        self.miniseed_buffer = {
            'ENZ': deque(maxlen=1500),  # 15 segundos a 100Hz
            'ENE': deque(maxlen=1500),
            'ENN': deque(maxlen=1500)
        }
        self.miniseed_timestamps = deque(maxlen=1500)
        
        # Hilos
        self.data_thread = None
        self.processing_thread = None
        
        logging.info("=== MONITOR CREIME_RT CONFIGURADO ===")
        logging.info(f"HOST: {host}:{port}")
        logging.info(f"VENTANA: {self.window_size} muestras ({self.window_size/sampling_rate} segundos)")
        logging.info(f"UMBRAL DETECCIÓN: {self.detection_threshold}")
        logging.info(f"UMBRAL RUIDO ALTO: {self.high_noise_threshold}")
        logging.info(f"VENTANAS CONSECUTIVAS: {self.consecutive_windows}")
    
    def connect_to_anyshake(self):
        """Conexión con AnyShake Observer"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                self.socket.connect((self.host, self.port))
                logging.info(f"✅ Conexión establecida: {self.host}:{self.port}")
                return True
                
            except Exception as e:
                logging.warning(f"Intento {attempt + 1}/{max_retries} falló: {e}")
                if self.socket:
                    self.socket.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error("No se pudo establecer conexión con AnyShake")
                    return False
    
    def parse_observer_packet(self, packet):
        """Parser optimizado para paquetes AnyShake"""
        try:
            parts = packet.strip().split(',')
            
            if len(parts) < 9 or parts[0][0] != '$' or '*' not in parts[-1]:
                return None
            
            component = parts[4]
            timestamp = int(parts[5])
            sampling_rate = int(parts[6])
            
            if sampling_rate != self.sampling_rate:
                return None
            
            data_start = 7
            data_end = len(parts) - 1
            raw_data = parts[data_start:data_end]
            
            data_values = [int(x) for x in raw_data]
            
            if data_values:
                data_values = self.hybrid_filter.apply_filter(data_values)
            
            return {
                'component': component,
                'timestamp': timestamp,
                'sampling_rate': sampling_rate,
                'data': data_values
            }
            
        except Exception as e:
            logging.debug(f"Error parseando paquete: {e}")
            return None
    
    def ultra_fast_processing(self):
        """Procesamiento ultra-rápido"""
        current_time = time.time()
        
        if current_time - self.last_processing_time < self.latency_target:
            return None
        
        window_data = self.buffer.get_latest_window()
        if window_data is None:
            return None
        
        processing_id = self.processing_count
        if self.processing_pipeline.submit_window(window_data, processing_id):
            self.processing_count += 1
            self.last_processing_time = current_time
            
            result_data = self.processing_pipeline.get_result(timeout=0.3)
            
            if result_data:
                processing_time = time.time() - current_time
                
                return {
                    'timestamp': datetime.now(),
                    'detection': result_data['result'][0],
                    'magnitude': result_data['result'][1],
                    'confidence': self._calculate_confidence(result_data),
                    'processing_id': processing_id,
                    'processing_time': processing_time,
                    'window_data': window_data
                }
        
        return None
    
    def _calculate_confidence(self, result_data):
        """Interpreta salida cruda de CREIME_RT"""
        try:
            if len(result_data['result']) >= 3:
                raw_output = float(result_data['result'][2])
            else:
                raw_output = self.noise_baseline
            
            # Mostrar timestamp actual en tiempo real
            current_time = datetime.now()
            logging.info(f"[{current_time}] CREIME_RT Raw Output: {raw_output:.2f}")
            
            return raw_output
            
        except (IndexError, TypeError, ValueError) as e:
            logging.warning(f"Error extrayendo salida CREIME_RT: {e}")
            current_time = datetime.now()
            logging.info(f"[{current_time}] CREIME_RT Raw Output: {self.noise_baseline:.2f} (error)")
            return self.noise_baseline
    
    def evaluate_detection(self, result):
        """Evaluación original CREIME_RT"""
        if result and result['confidence'] > self.detection_threshold:
            # Clasificación original CREIME_RT
            is_seismic = result['confidence'] > self.magnitude_threshold
            
            logging.info(f"DETECCIÓN: Confianza {result['confidence']:.2f} > {self.detection_threshold}")
            
            return {
                'type': 'event_detected',
                'consecutive_detections': 1,
                'is_seismic': is_seismic
            }
        
        return False
    
    def _apply_magnitude_correction(self, raw_output):
        """Interpretación original CREIME_RT - sin corrección"""
        return raw_output
    
    def _is_seismic_event(self, result):
        """Clasificación original CREIME_RT"""
        return result['confidence'] > self.magnitude_threshold
    
    def trigger_alert(self, detection_result, detection_info):
        """Activa alerta según protocolo oficial CREIME_RT"""
        self.detection_count += 1
        self.last_detection_time = detection_result['timestamp']
        
        # Calcular tiempo real del evento
        event_time = detection_result['timestamp']
        
        # Registrar primera detección
        if self.first_detection_time is None:
            self.first_detection_time = event_time
        
        if detection_info['is_seismic']:
            # Registrar primera confirmación de sismo
            if self.first_confirmation_time is None:
                self.first_confirmation_time = event_time
            
            # Aplicar corrección de magnitud para sismos confirmados
            raw_confidence = detection_result['confidence']
            corrected_magnitude = self._apply_magnitude_correction(raw_confidence)
            
            alert_message = (
                f"🚨 MONITOR: SISMO CONFIRMADO 🚨\n"
                f"Salida CREIME_RT: {detection_result['confidence']:.2f}\n"
                f"Magnitud: {corrected_magnitude:.1f}\n"
                f"Ventanas consecutivas: {detection_info['consecutive_detections']}/{self.consecutive_windows}\n"
                f"Ventana: {detection_result['processing_id']}\n"
                f"Latencia: {detection_result['processing_time']:.3f}s"
            )
            logging.critical(alert_message)
            
            # Registrar evento detectado con magnitud corregida
            self.detected_events.append({
                'type': 'seismic',
                'event_time': event_time,
                'confidence': detection_result['confidence'],
                'magnitude': detection_result['magnitude'],
                'corrected_magnitude': corrected_magnitude,
                'processing_id': detection_result['processing_id']
            })
            
            self.save_event_data(detection_result, corrected_magnitude)
        else:
            mag_display = f"{detection_result['magnitude']:.1f}" if detection_result['magnitude'] is not None else "N/A"
            alert_message = (
                f"⚠️ MONITOR: EVENTO DETECTADO ⚠️\n"
                f"Salida CREIME_RT: {detection_result['confidence']:.2f}\n"
                f"Magnitud: {mag_display}\n"
                f"Ventanas consecutivas: {detection_info['consecutive_detections']}/{self.consecutive_windows}\n"
                f"Ventana: {detection_result['processing_id']}\n"
                f"Latencia: {detection_result['processing_time']:.3f}s"
            )
            logging.warning(alert_message)
            
            # Registrar evento detectado
            self.detected_events.append({
                'type': 'event',
                'event_time': event_time,
                'confidence': detection_result['confidence'],
                'magnitude': detection_result['magnitude'],
                'processing_id': detection_result['processing_id']
            })
    
    def save_event_data(self, detection_result, corrected_magnitude):
        """Guarda datos del evento detectado en monitor"""
        try:
            events_dir = "events_monitor"
            if not os.path.exists(events_dir):
                os.makedirs(events_dir)
            
            event_id = str(uuid.uuid4())[:8]
            timestamp_str = detection_result['timestamp'].strftime('%Y%m%d_%H%M%S')
            
            # 1. Guardar JSON inmediatamente
            json_data = {
                "station_id": self.station_id,
                "event_id": event_id,
                "timestamp": detection_result['timestamp'].isoformat(),
                "confidence": detection_result['confidence'],
                "magnitude": detection_result['magnitude'],
                "corrected_magnitude": corrected_magnitude
            }
            
            json_filename = os.path.join(events_dir, f"monitor_event_{timestamp_str}.json")
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            logging.info(f"Evento monitor guardado: {json_filename}")
            
            # 2. Programar MiniSEED para 60 segundos después
            miniseed_timer = threading.Timer(60.0, self._save_miniseed, 
                                           args=[event_id, timestamp_str, detection_result['timestamp']])
            miniseed_timer.daemon = True
            miniseed_timer.start()
            logging.info(f"MiniSEED programado para 60s: monitor_event_{timestamp_str}.mseed")
            
        except Exception as e:
            logging.error(f"Error guardando evento monitor: {e}")
    
    def _save_miniseed(self, event_id, timestamp_str, event_time):
        """Guarda archivo MiniSEED con 1 minuto de datos (15s antes del evento)"""
        try:
            events_dir = "events_monitor"
            
            # Crear stream ObsPy
            stream = Stream()
            
            # Obtener datos de los últimos 60 segundos (6000 muestras)
            for i, component in enumerate(['ENZ', 'ENE', 'ENN']):
                channel_code = ['HHZ', 'HHE', 'HHN'][i]
                
                # Extraer últimas 6000 muestras (60 segundos)
                buffer_data = list(self.miniseed_buffer[component])[-6000:] if len(self.miniseed_buffer[component]) >= 6000 else list(self.miniseed_buffer[component])
                
                if len(buffer_data) < 6000:
                    # Rellenar con ceros si no hay suficientes datos
                    buffer_data = [0.0] * (6000 - len(buffer_data)) + buffer_data
                
                # Crear trace
                stats = Stats()
                stats.network = "SK"
                stats.station = "MONITOR"
                stats.location = "00"
                stats.channel = channel_code
                stats.sampling_rate = 100.0
                stats.starttime = UTCDateTime(event_time) - 15  # 15 segundos antes del evento
                
                trace = Trace(data=np.array(buffer_data, dtype=np.float32), header=stats)
                stream.append(trace)
            
            # Guardar MiniSEED
            mseed_filename = os.path.join(events_dir, f"monitor_event_{timestamp_str}.mseed")
            stream.write(mseed_filename, format='MSEED')
            logging.info(f"MiniSEED guardado: {mseed_filename}")
            
        except Exception as e:
            logging.error(f"Error guardando MiniSEED: {e}")
    
    def processing_loop(self):
        """Bucle de procesamiento con estabilidad 24/7"""
        self.last_processing_time = time.time()
        cycle_count = 0
        
        while self.running:
            try:
                if self.buffer.wait_for_new_data(timeout=0.3):
                    self.buffer.reset_data_event()
                    
                    result = self.ultra_fast_processing()
                    
                    if result:
                        # Capturar valor y timestamp para estadísticas
                        self.creime_values.append(result['confidence'])
                        self.creime_timestamps.append(result['timestamp'])
                        
                        detection_info = self.evaluate_detection(result)
                        if detection_info:
                            self.trigger_alert(result, detection_info)
                
                # Limpieza periódica de memoria (cada hora)
                cycle_count += 1
                if cycle_count % 3600 == 0:  # 3600 segundos = 1 hora
                    self._memory_cleanup()
                    logging.info(f"Limpieza de memoria completada - Ciclo {cycle_count}")
                
                time.sleep(0.05)
                
            except Exception as e:
                logging.error(f"Error en bucle procesamiento monitor: {e}")
                self._handle_processing_error()
                time.sleep(0.1)
    
    def receive_data_loop(self):
        """Bucle de recepción de datos AnyShake"""
        self.running = True
        self.data_buffer = b''
        
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    logging.warning("Conexión cerrada - Reconectando...")
                    if not self.connect_to_anyshake():
                        time.sleep(3)
                        continue
                    else:
                        continue
                
                self.data_buffer += data
                
                while b'\r' in self.data_buffer:
                    packet, self.data_buffer = self.data_buffer.split(b'\r', 1)
                    
                    try:
                        packet_str = packet.decode('ascii', errors='ignore').strip()
                        if packet_str:
                            parsed_data = self.parse_observer_packet(packet_str)
                            if parsed_data:
                                current_time = time.time()
                                
                                self.buffer.add_data(
                                    parsed_data['component'],
                                    parsed_data['data'],
                                    current_time
                                )
                                
                                # Actualizar buffer MiniSEED
                                if parsed_data['component'] in self.miniseed_buffer:
                                    self.miniseed_buffer[parsed_data['component']].extend(parsed_data['data'])
                                    for _ in parsed_data['data']:
                                        self.miniseed_timestamps.append(current_time)
                                
                                self.visualizer.update_data(
                                    parsed_data['component'],
                                    parsed_data['data'],
                                    current_time
                                )
                                
                                self.packet_count += 1
                                
                    except Exception as e:
                        continue
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error en recepción: {e}")
                    time.sleep(3)
                else:
                    break
    
    def plot_creime_output_timeline(self):
        """Genera gráfico de raw output CREIME_RT vs tiempo"""
        if not self.creime_values or not VISUALIZATION_ENABLED:
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Convertir timestamps a datetime
            if self.creime_timestamps:
                times = [ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00')) 
                        for ts in self.creime_timestamps]
            else:
                times = list(range(len(self.creime_values)))
            
            # Crear gráfico
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Plotear raw output
            ax.plot(times, self.creime_values, 'b-', linewidth=1.0, alpha=0.8, label='CREIME_RT Raw Output')
            
            # Líneas de umbrales
            ax.axhline(y=self.detection_threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Umbral Detección ({self.detection_threshold})')
            ax.axhline(y=self.noise_baseline, color='gray', linestyle='--', linewidth=1, 
                      label=f'Línea Base Ruido ({self.noise_baseline})')
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1, 
                      label='Cero (Magnitud Positiva)')
            ax.axhline(y=self.magnitude_threshold, color='orange', linestyle='--', linewidth=1, 
                      label=f'Umbral Magnitud ({self.magnitude_threshold})')
            
            # Configuración del gráfico
            ax.set_xlabel('Tiempo', fontsize=12)
            ax.set_ylabel('CREIME_RT Raw Output', fontsize=12)
            ax.set_title('Evolución Temporal CREIME_RT - Monitor en Tiempo Real', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Formatear eje X si son timestamps reales
            if self.creime_timestamps and isinstance(self.creime_timestamps[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Estadísticas en el gráfico
            stats_text = (
                f'Valores: {len(self.creime_values)}\n'
                f'Mínimo: {min(self.creime_values):.3f}\n'
                f'Máximo: {max(self.creime_values):.3f}\n'
                f'Promedio: {sum(self.creime_values)/len(self.creime_values):.3f}\n'
                f'Detecciones: {sum(1 for v in self.creime_values if v > self.detection_threshold)}'
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            plt.show()
            
            logging.info("Gráfico de raw output generado")
            
        except Exception as e:
            logging.error(f"Error generando gráfico: {e}")
    
    def start_monitor(self):
        """Inicia el monitor completo"""
        if not self.connect_to_anyshake():
            return False
        
        self.start_time = time.time()
        self.monitor_start_time = datetime.now()
        
        logging.info("INICIANDO MONITOR CREIME_RT")
        logging.info(f" Host: {self.host}:{self.port}")
        logging.info(f" Umbral Detección: {self.detection_threshold}")
        logging.info(f" Ventanas Consecutivas: {self.consecutive_windows}")
        logging.info(f" Tiempo inicio: {self.monitor_start_time}")
        
        self.running = True
        
        # Hilo de recepción de datos
        self.data_thread = threading.Thread(
            target=self.receive_data_loop,
            name="AnyShakeReceiver",
            daemon=True
        )
        self.data_thread.start()
        
        # Esperar a que el worker CREIME_RT esté completamente inicializado
        logging.info("Esperando inicialización completa del worker CREIME_RT...")
        
        if self.processing_pipeline.is_worker_ready(timeout=30):
            logging.info("Worker CREIME_RT inicializado correctamente")
            
            # Hilo de procesamiento
            self.processing_thread = threading.Thread(
                target=self.processing_loop,
                name="MonitorProcessor", 
                daemon=True
            )
            self.processing_thread.start()
            
            # Iniciar visualizador solo si el worker está listo
            if VISUALIZATION_ENABLED:
                logging.info("Iniciando visualizador - Worker CREIME_RT operativo")
                self.visualizer.start_visualization()
            else:
                logging.info("Visualización desactivada - Monitor en modo consola")
                
        else:
            logging.error("Worker CREIME_RT no se inicializó correctamente")
            logging.error("Visualizador NO iniciado - Sistema en modo degradado")
            return False
        
        # Esperar inicialización
        buffer_ready = False
        startup_time = time.time()
        while not buffer_ready and self.running and (time.time() - startup_time < 10):
            time.sleep(0.1)
            status = self.buffer.get_buffer_status()
            current_samples = status['ENZ']['samples']
            buffer_ready = current_samples >= 50
            
            if not buffer_ready and (time.time() - startup_time > 1):
                elapsed = time.time() - startup_time
                logging.info(f"Inicializando buffer monitor: {current_samples}/3000 muestras ({elapsed:.1f}s)")
        
        if buffer_ready:
            logging.info("MONITOR OPERATIVO - Procesando datos en tiempo real")
        else:
            logging.warning("Monitor operativo con buffer parcial")
        
        return True
    
    def _memory_cleanup(self):
        """Limpieza periódica de memoria para operación 24/7"""
        try:
            # Limpiar buffers de estadísticas si son muy grandes
            if len(self.creime_values) > 10000:
                self.creime_values = self.creime_values[-5000:]  # Mantener últimos 5000
                self.creime_timestamps = self.creime_timestamps[-5000:]
            
            # Garbage collection forzado
            import gc
            gc.collect()
            
        except Exception as e:
            logging.warning(f"Error en limpieza de memoria: {e}")
    
    def _handle_processing_error(self):
        """Manejo de errores de procesamiento para recuperación automática"""
        try:
            # Intentar reinicializar pipeline si hay errores consecutivos
            logging.warning("Intentando recuperación automática del pipeline")
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error en recuperación automática: {e}")
    
    def stop_monitor(self):
        """Detiene el monitor con limpieza completa"""
        logging.info("Iniciando parada segura del monitor...")
        self.running = False
        
        # Parada ordenada de componentes
        try:
            self.processing_pipeline.stop_workers()
        except Exception as e:
            logging.warning(f"Error deteniendo workers: {e}")
        
        try:
            self.visualizer.stop_visualization()
        except Exception as e:
            logging.warning(f"Error deteniendo visualizador: {e}")
        
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logging.warning(f"Error cerrando socket: {e}")
        
        # Limpieza final de memoria
        self._memory_cleanup()
        
        if self.start_time:
            run_time = time.time() - self.start_time
            processing_rate = self.processing_count / run_time if run_time > 0 else 0
            
            logging.info(f"\n{'='*60}")
            logging.info(f"REPORTE FINAL - MONITOR CREIME_RT")
            logging.info(f"{'='*60}")
            logging.info(f"Monitor en Tiempo Real:")
            logging.info(f"  Inicio: {self.monitor_start_time}")
            logging.info(f"  Duración: {run_time:.1f} segundos")
            logging.info(f"")
            logging.info(f"Rendimiento del Sistema:")
            logging.info(f"  Paquetes procesados: {self.packet_count}")
            logging.info(f"  Ventanas CREIME_RT: {self.processing_count}")
            logging.info(f"  Tasa procesamiento: {processing_rate:.2f} ventanas/segundo")
            logging.info(f"")
            
            # Tiempos de detección
            if self.first_detection_time:
                logging.info(f"Primera Detección: {self.first_detection_time}")
            if self.first_confirmation_time:
                logging.info(f"Primera Confirmación: {self.first_confirmation_time}")
            
            logging.info(f"Eventos Detectados: {len(self.detected_events)}")
            
            # Estadísticas de valores CREIME_RT para diagnóstico
            if hasattr(self, 'creime_values') and self.creime_values:
                min_val = min(self.creime_values)
                max_val = max(self.creime_values)
                mean_val = sum(self.creime_values) / len(self.creime_values)
                above_threshold = sum(1 for v in self.creime_values if v > self.detection_threshold)
                logging.info(f"")
                logging.info(f"Estadísticas CREIME_RT:")
                logging.info(f"  Valores mínimo/máximo: {min_val:.2f} / {max_val:.2f}")
                logging.info(f"  Valor promedio: {mean_val:.2f}")
                logging.info(f"  Ventanas > umbral ({self.detection_threshold}): {above_threshold}/{len(self.creime_values)}")
                logging.info(f"  Porcentaje activación: {(above_threshold/len(self.creime_values)*100):.2f}%")
            
            if self.detected_events:
                # Calcular magnitud máxima
                max_magnitude = 0.0
                for event in self.detected_events:
                    if event['type'] == 'seismic' and 'corrected_magnitude' in event:
                        max_magnitude = max(max_magnitude, event['corrected_magnitude'])
                
                logging.info(f"Magnitud Final (Máxima): {max_magnitude:.1f}")
                logging.info(f"")
                logging.info(f"Detalle de Eventos:")
                for i, event in enumerate(self.detected_events, 1):
                    event_type = "SÍSMICO" if event['type'] == 'seismic' else "EVENTO"
                    logging.info(f"  {i}. {event_type}:")
                    logging.info(f"     Tiempo: {event['event_time']}")
                    logging.info(f"     Confianza: {event['confidence']:.2f}")
                    if 'corrected_magnitude' in event:
                        logging.info(f"     Magnitud: {event['corrected_magnitude']:.1f}")
                    else:
                        logging.info(f"     Magnitud: {event['magnitude']:.1f if event['magnitude'] else 'N/A'}")
                    logging.info(f"     Ventana: {event['processing_id']}")
            else:
                logging.info(f"  No se detectaron eventos sísmicos")
            
            logging.info(f"{'='*60}")
            
            # Generar gráfico de raw output
            self.plot_creime_output_timeline()

def main():
    """Función principal del monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor CREIME_RT en tiempo real con AnyShake')
    parser.add_argument('--model_path', default='../saipy/saved_models/', help='Ruta del modelo CREIME_RT')
    parser.add_argument('--host', default='localhost', help='Host de AnyShake Observer')
    parser.add_argument('--port', type=int, default=30000, help='Puerto de AnyShake Observer')
    
    args = parser.parse_args()
    
    # Crear directorios
    for directory in ["logs", "events_monitor"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Verificar SAIPy
    try:
        import saipy
        logging.info("SAIPy disponible para monitor")
    except ImportError:
        logging.error("SAIPy no disponible")
        sys.exit(1)
    
    # Crear monitor
    monitor = RealTimeMonitor(
        model_path=args.model_path,
        host=args.host,
        port=args.port
    )
    
    try:
        logging.info(f"Iniciando monitor CREIME_RT en tiempo real")
        logging.info("Presiona Ctrl+C para detener el sistema")
        
        if monitor.start_monitor():
            if VISUALIZATION_ENABLED:
                plt.show(block=True)
            else:
                while monitor.running:
                    time.sleep(1)
        else:
            logging.error("Fallo en inicio del monitor")
            
    except KeyboardInterrupt:
        logging.info("\nMonitor detenido por usuario (Ctrl+C)")
    except Exception as e:
        logging.error(f"Error crítico en monitor: {e}")
    finally:
        monitor.stop_monitor()

if __name__ == "__main__":
    logging.info("Ejecutando Monitor CREIME_RT en Tiempo Real")
    main()