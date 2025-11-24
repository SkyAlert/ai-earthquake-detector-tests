#!/usr/bin/env python3
"""
SIMULADOR CREIME_RT - Sistema de Alerta Sísmica con Datos MiniSEED
Simulador que reproduce el comportamiento de CREIME_RT_Monitor usando archivos MiniSEED
"""

import os
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
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import Stats

# ===== CONFIGURACIÓN GPU SEGURA PARA JETSON ORIN NANO =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configuración de logging optimizada
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "creime_rt_simulator.log")

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
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import style
    style.use('fivethirtyeight')
    plt.figure()
    plt.close()
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

# Constantes de conversión (ya aplicadas en MiniSEED)
SENSITIVITY = 1.0  # Los datos MiniSEED ya están en unidades físicas
MG_TO_GALS = 1.0
CONVERSION_FACTOR = SENSITIVITY * MG_TO_GALS

# Colores personalizados
COLOR_TEAL = '#009688'
COLOR_ORANGE = '#FF9800'
COLOR_RED = '#FF0000'
COLOR_GREEN = '#00FF00'

class RealTimeVisualizer:
    """Sistema de visualización para simulador MiniSEED"""
    
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
        """Configuración de visualización para simulador"""
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
            
            self.fig.suptitle('CREIME_RT SIMULATOR - Datos Procesados CREIME_RT', 
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
                ax.set_ylabel('Amplitud Filtrada [1-45Hz]', fontsize=12)
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
            logging.info("Visualizador del simulador configurado correctamente")
            
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
        """Algoritmo mejorado para escalado dinámico con mejor resolución"""
        if len(data) == 0:
            return -1, 1
        
        # Usar percentiles para mejor escalado
        p95 = np.percentile(np.abs(data), 95)
        p99 = np.percentile(np.abs(data), 99)
        
        self.max_values_history[component_name].append(p95)
        
        if len(self.max_values_history[component_name]) > 0:
            # Usar percentil 75 del historial para estabilidad
            historical_max = np.percentile(list(self.max_values_history[component_name]), 75)
            target_max = max(p99, historical_max)
        else:
            target_max = p99
        
        # Margen más ajustado para mejor resolución
        margin = 1.1
        limit = target_max * margin
        
        # Límite mínimo más pequeño para señales débiles
        min_limit = max(0.01, target_max * 0.1)
        
        if limit < min_limit:
            limit = min_limit
        
        return -limit, limit
    
    def update_data(self, component, data, timestamp):
        """Actualización de datos para visualización"""
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
            return [self.line_enz, self.line_ene, self.line_enn, self.info_text]
        
        with self.lock:
            times_copy = np.array(self.times)
            enz_copy = np.array(self.data_enz)
            ene_copy = np.array(self.data_ene)
            enn_copy = np.array(self.data_enn)
        
        min_len = min(len(times_copy), len(enz_copy), len(ene_copy), len(enn_copy))
        if min_len < 10:
            artists = [self.line_enz, self.line_ene, self.line_enn, self.info_text]
            for markers in self.processing_window_markers:
                artists.extend([markers['line'], markers['fill']])
            return artists
        
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
        
        artists = [self.line_enz, self.line_ene, self.line_enn, self.info_text]
        for markers in self.processing_window_markers:
            artists.extend([markers['line'], markers['fill']])
        return artists
    
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
        """Inicia visualización"""
        if not self.visualization_enabled:
            return
            
        try:
            self.running = True
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=150, blit=True, cache_frame_data=False
            )
            logging.info("Visualizador del simulador iniciado")
        except Exception as e:
            logging.error(f"Error iniciando visualización: {e}")
            self.visualization_enabled = False
    
    def stop_visualization(self):
        """Detiene el sistema de visualización"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        logging.info("Visualizador detenido")

class UltraFastBuffer:
    """Buffer ultra-rápido para simulador MiniSEED"""
    
    def __init__(self, window_size=3000, sampling_rate=100, update_interval=0.1):
        self.window_size = int(window_size)
        self.sampling_rate = int(sampling_rate)
        self.update_interval = update_interval
        
        total_size = self.window_size + 500  # Buffer para 35 segundos
        self.buffers = {
            'ENZ': deque(maxlen=total_size),
            'ENE': deque(maxlen=total_size),
            'ENN': deque(maxlen=total_size)
        }
        
        self.lock = threading.Lock()
        self.window_count = 0
        self.last_window_time = 0
        self.ready = False
        self.min_ready_samples = int(30 * sampling_rate)  # 30 segundos mínimos
        self.new_data_event = threading.Event()
        
        self.performance_stats = {
            'windows_generated': 0,
            'data_points_received': 0,
            'last_update': time.time()
        }
    
    def add_data(self, component, data, timestamp):
        """Añade datos del simulador MiniSEED"""
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
        """Genera ventana deslizante"""
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_window_time < self.update_interval:
                return None
                
            if not self.ready:
                return None
            
            window_data = []
            for component in ['ENZ', 'ENE', 'ENN']:
                buf = self.buffers[component]
                buf_len = len(buf)
                
                if buf_len < self.window_size:
                    padding_needed = self.window_size - buf_len
                    component_data = [0.0] * padding_needed + list(buf)
                else:
                    start_idx = buf_len - self.window_size
                    component_data = [buf[i] for i in range(start_idx, buf_len)]
                
                # Aplicar solo normalización z-score (datos ya vienen filtrados)
                if len(component_data) > 1:
                    mean_val = np.mean(component_data)
                    std_val = np.std(component_data)
                    if std_val > 0:
                        component_data = [(x - mean_val) / std_val for x in component_data]
                
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
    
    def _apply_bandpass_filter(self, data):
        """Aplica filtro pasa-banda [1.0, 45.0] Hz"""
        try:
            from scipy.signal import butter, filtfilt
            fs = self.sampling_rate
            nyquist = 0.5 * fs
            low = 1.0 / nyquist
            high = 45.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, data).tolist()
        except:
            return data

class OptimizedHybridFilter:
    """Filtro optimizado según documentación CREIME_RT"""
    
    def __init__(self, fs=100, hp_cutoff=1.0, lp_cutoff=45):
        self.fs = fs
        nyquist = 0.5 * fs
        
        normal_cutoff_hp = hp_cutoff / nyquist
        self.b_hp, self.a_hp = butter(2, normal_cutoff_hp, btype='high', analog=False)
        
        normal_cutoff_lp = lp_cutoff / nyquist
        self.b_lp, self.a_lp = butter(4, normal_cutoff_lp, btype='low', analog=False)
        
        self.zi_hp = np.zeros(max(len(self.a_hp), len(self.b_hp)) - 1)
        self.zi_lp = np.zeros(max(len(self.a_lp), len(self.b_lp)) - 1)
        
        self.detrend_buffer = deque(maxlen=fs)
    
    def apply_filter(self, data):
        """Aplica filtro optimizado"""
        if not data:
            return data
        
        self.detrend_buffer.extend(data)
        if len(self.detrend_buffer) > 0:
            midpoint = np.median(list(self.detrend_buffer)).astype(np.float32)
        else:
            midpoint = 0.0
            
        detrended = [x - midpoint for x in data]
        detrended_np = np.array(detrended, dtype=np.float32)
        
        filtered_hp, self.zi_hp = lfilter(self.b_hp, self.a_hp, detrended_np, zi=self.zi_hp)
        filtered_lp, self.zi_lp = lfilter(self.b_lp, self.a_lp, filtered_hp, zi=self.zi_lp)
        
        return filtered_lp.astype(np.float32).tolist()

class UltraFastProcessingPipeline:
    """Pipeline de procesamiento para simulador"""
    
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
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self.processing_worker,
                name=f"CREIME_Simulator_Worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Iniciados {self.num_workers} workers del simulador")
    
    def processing_worker(self):
        """Worker para CREIME_RT en simulador"""
        try:
            from saipy.models.creime import CREIME_RT
            model = CREIME_RT(self.model_path)
            logging.info("Worker simulador CREIME_RT inicializado")
        except Exception as e:
            logging.error(f"Worker simulador no pudo cargar modelo: {e}")
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
                    
                    if raw_output > -1.80:
                        detection = 1
                        magnitude = max(raw_output, 0.0) if raw_output > 0 else None
                    else:
                        detection = 0
                        magnitude = None
                    
                    result = (detection, magnitude, raw_output)
                    
                except (IndexError, ValueError, TypeError) as e:
                    logging.warning(f"Error en predicción simulador: {e}")
                    result = (0, None, -4.0)
                except Exception as e:
                    logging.error(f"Error inesperado en simulador: {e}")
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
                logging.error(f"Error en worker simulador: {e}")
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

class MiniSeedSimulator:
    """
    Simulador principal que reproduce CREIME_RT_Monitor usando archivos MiniSEED
    LONGITUD MÍNIMA REQUERIDA: 15 segundos (1500 muestras por componente)
    """
    
    def __init__(self, miniseed_file, model_path, sampling_rate=100, playback_speed=1.0):
        self.miniseed_file = miniseed_file
        self.model_path = model_path
        self.sampling_rate = sampling_rate
        self.playback_speed = playback_speed  # 1.0 = tiempo real, 2.0 = 2x velocidad
        
        # Parámetros según documentación oficial CREIME_RT
        self.window_size = 30 * sampling_rate  # 3000 muestras - 30 SEGUNDOS (oficial)
        self.latency_target = 0.1 / playback_speed  # Ajustado por velocidad
        self.detection_threshold = -1.80  # Nuevo umbral de detección
        self.noise_baseline = -4.0
        self.high_noise_threshold = -1.80  # Umbral para ruido alto
        self.magnitude_threshold = 1.0  # Ajustado según análisis (máximo 1.6)
        self.consecutive_windows = 2  # Cambiado a 2 ventanas consecutivas
        
        # Componentes del sistema
        self.buffer = UltraFastBuffer(
            window_size=self.window_size,
            sampling_rate=sampling_rate,
            update_interval=self.latency_target
        )
        
        self.hybrid_filter = OptimizedHybridFilter(fs=sampling_rate)
        self.processing_pipeline = UltraFastProcessingPipeline(model_path, num_workers=1)
        self.visualizer = RealTimeVisualizer(self)
        
        # Estado del sistema
        self.running = False
        self.stream_data = None
        
        # Estadísticas
        self.detection_count = 0
        self.last_detection_time = None
        self.packet_count = 0
        self.start_time = None
        self.processing_count = 0
        self.last_processing_time = 0
        self.detection_buffer = deque(maxlen=self.consecutive_windows)
        
        # Configuración de estación
        self.station_id = "CREIME_RT_SIMULATOR"
        
        # Rastreo de eventos y timestamps
        self.detected_events = []
        self.miniseed_start_time = None
        self.miniseed_duration = 0
        self.simulation_start_time = None
        
        # Rastreo de tiempos de detección
        self.first_detection_time = None  # Primer evento detectado
        self.first_confirmation_time = None  # Primera confirmación de sismo
        
        # Rastreo de valores CREIME_RT para diagnóstico
        self.creime_values = []
        self.creime_timestamps = []  # Timestamps MiniSEED correspondientes
        
        # Hilos
        self.data_thread = None
        self.processing_thread = None
        
        logging.info("=== SIMULADOR CREIME_RT CONFIGURADO ===")
        logging.info(f"ARCHIVO MINISEED: {miniseed_file}")
        logging.info(f"VELOCIDAD REPRODUCCIÓN: {playback_speed}x")
        logging.info(f"VENTANA: {self.window_size} muestras ({self.window_size/sampling_rate} segundos)")
        logging.info(f"UMBRAL DETECCIÓN: {self.detection_threshold} (oficial CREIME_RT)")
        logging.info(f"UMBRAL RUIDO ALTO: {self.high_noise_threshold}")
        logging.info(f"VENTANAS CONSECUTIVAS: {self.consecutive_windows}")
    
    def load_miniseed_data(self):
        """Carga y valida archivo MiniSEED"""
        try:
            logging.info(f"Cargando archivo MiniSEED: {self.miniseed_file}")
            stream = read(self.miniseed_file)
            
            # Validar componentes requeridos
            required_channels = ['HHZ', 'HHE', 'HHN']
            available_channels = [tr.stats.channel for tr in stream]
            
            logging.info(f"Canales disponibles: {available_channels}")
            
            # Mapear canales a componentes estándar
            channel_mapping = {}
            for tr in stream:
                channel = tr.stats.channel
                # Mapeo más robusto de canales
                if 'Z' in channel.upper():
                    channel_mapping['ENZ'] = tr
                    logging.info(f"Canal Z mapeado: {channel} -> ENZ")
                elif 'E' in channel.upper() or '1' in channel or 'EW' in channel.upper():
                    channel_mapping['ENE'] = tr
                    logging.info(f"Canal E mapeado: {channel} -> ENE")
                elif 'N' in channel.upper() or '2' in channel or 'NS' in channel.upper():
                    channel_mapping['ENN'] = tr
                    logging.info(f"Canal N mapeado: {channel} -> ENN")
            
            if len(channel_mapping) < 3:
                raise ValueError(f"Se requieren 3 componentes, encontrados: {len(channel_mapping)}")
            
            # Validar longitud mínima (15 segundos)
            min_length = 15 * self.sampling_rate  # 1500 muestras
            for comp, tr in channel_mapping.items():
                if len(tr.data) < min_length:
                    raise ValueError(f"Componente {comp}: {len(tr.data)} muestras < {min_length} mínimas (15s)")
            
            # Validar y ajustar frecuencia de muestreo
            for comp, tr in channel_mapping.items():
                actual_rate = tr.stats.sampling_rate
                if abs(actual_rate - self.sampling_rate) > 0.1:
                    logging.warning(f"Frecuencia {comp}: {actual_rate} Hz, esperada: {self.sampling_rate} Hz")
                    # Resamplear si es necesario
                    if actual_rate != self.sampling_rate:
                        tr.resample(self.sampling_rate)
                        logging.info(f"Remuestreado {comp} de {actual_rate} Hz a {self.sampling_rate} Hz")
            
            self.stream_data = channel_mapping
            
            # Información detallada del archivo
            total_samples = min(len(tr.data) for tr in channel_mapping.values())
            duration = total_samples / self.sampling_rate
            
            # Capturar tiempo de inicio del MiniSEED
            first_trace = list(channel_mapping.values())[0]
            self.miniseed_start_time = first_trace.stats.starttime
            self.miniseed_duration = duration
            
            logging.info(f"✅ MiniSEED cargado exitosamente:")
            logging.info(f"   Duración: {duration:.1f} segundos ({total_samples} muestras)")
            logging.info(f"   Componentes: {list(channel_mapping.keys())}")
            logging.info(f"   Frecuencia: {self.sampling_rate} Hz")
            
            # Mostrar estadísticas de cada componente
            for comp, tr in channel_mapping.items():
                data_min = np.min(tr.data)
                data_max = np.max(tr.data)
                data_mean = np.mean(tr.data)
                data_std = np.std(tr.data)
                logging.info(f"   {comp}: min={data_min:.6f}, max={data_max:.6f}, mean={data_mean:.6f}, std={data_std:.6f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error cargando MiniSEED: {e}")
            return False
    
    def simulate_data_stream(self):
        """Simula el flujo de datos en tiempo real"""
        if not self.stream_data:
            logging.error("No hay datos MiniSEED cargados")
            return
        
        logging.info("Iniciando simulación de flujo de datos...")
        
        # Obtener longitud mínima
        total_samples = min(len(tr.data) for tr in self.stream_data.values())
        samples_per_packet = 10  # 10 muestras por paquete (0.1 segundos) - precisión original
        
        sample_index = 0
        packet_interval = (samples_per_packet / self.sampling_rate) / self.playback_speed
        start_simulation_time = time.time()
        self.simulation_start_time = start_simulation_time
        
        logging.info(f"Iniciando simulación: {total_samples} muestras, intervalo {packet_interval:.4f}s")
        
        while self.running and sample_index < total_samples - samples_per_packet:
            try:
                # Extraer datos para este paquete
                for component in ['ENZ', 'ENE', 'ENN']:
                    if component in self.stream_data:
                        tr = self.stream_data[component]
                        packet_data = tr.data[sample_index:sample_index + samples_per_packet]
                        
                        # Aplicar el mismo procesamiento que usará CREIME_RT
                        # 1. Filtrado [1.0, 45.0] Hz
                        try:
                            from scipy.signal import butter, filtfilt
                            fs = self.sampling_rate
                            nyquist = 0.5 * fs
                            low = 1.0 / nyquist
                            high = 45.0 / nyquist
                            b, a = butter(4, [low, high], btype='band')
                            filtered_data = filtfilt(b, a, packet_data).tolist()
                        except:
                            filtered_data = packet_data.astype(np.float32).tolist()
                        
                        unified_data = filtered_data
                        
                        # Diagnóstico para primeros paquetes
                        if self.packet_count < 3:
                            logging.info(f"Paquete {self.packet_count} {component}:")
                            logging.info(f"  MiniSEED Original: {packet_data[:3]}... (rango: {np.min(packet_data):.6f} a {np.max(packet_data):.6f})")
                            logging.info(f"  Filtrado [1-45Hz]: {unified_data[:3]}... (rango: {np.min(unified_data):.6f} a {np.max(unified_data):.6f})")
                            logging.info(f"  Visualización = CREIME_RT (mismo procesamiento)")
                        
                        # Usar los mismos datos procesados para buffer y visualizador
                        current_time = time.time()
                        self.buffer.add_data(component, unified_data, current_time)
                        
                        # Visualizador muestra exactamente lo que ve CREIME_RT
                        self.visualizer.update_data(component, unified_data, current_time)
                
                self.packet_count += 1
                sample_index += samples_per_packet
                
                # Progreso y diagnóstico
                if self.packet_count % 100 == 0:
                    progress = (sample_index / total_samples) * 100
                    elapsed = sample_index / self.sampling_rate
                    real_elapsed = time.time() - start_simulation_time
                    
                    # Mostrar muestra de datos para diagnóstico
                    if self.packet_count == 100:
                        for comp in ['ENZ', 'ENE', 'ENN']:
                            if comp in self.stream_data:
                                sample_data = self.stream_data[comp].data[sample_index:sample_index+5]
                                logging.info(f"Muestra {comp}[{sample_index}:{sample_index+5}]: {sample_data}")
                    
                    logging.info(f"Progreso: {progress:.1f}% - Sim: {elapsed:.1f}s - Real: {real_elapsed:.1f}s - Velocidad: {elapsed/real_elapsed:.2f}x")
                
                # Esperar según velocidad de reproducción
                time.sleep(packet_interval)
                
            except Exception as e:
                logging.error(f"Error en simulación de datos: {e}")
                break
        
        logging.info("Simulación de datos completada")
    
    def ultra_fast_processing(self):
        """Procesamiento ultra-rápido (igual que el original)"""
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
            
            # Calcular tiempo MiniSEED correspondiente
            if self.miniseed_start_time and self.simulation_start_time:
                elapsed_simulation = time.time() - self.simulation_start_time
                miniseed_time = self.miniseed_start_time + elapsed_simulation
                logging.info(f"[{miniseed_time}] CREIME_RT Raw Output: {raw_output:.6f}")
            else:
                logging.info(f"CREIME_RT Raw Output: {raw_output:.6f}")
            
            return raw_output
            
        except (IndexError, TypeError, ValueError) as e:
            logging.warning(f"Error extrayendo salida CREIME_RT: {e}")
            if self.miniseed_start_time and self.simulation_start_time:
                elapsed_simulation = time.time() - self.simulation_start_time
                miniseed_time = self.miniseed_start_time + elapsed_simulation
                logging.info(f"[{miniseed_time}] CREIME_RT Raw Output: {self.noise_baseline:.6f} (error)")
            else:
                logging.info(f"CREIME_RT Raw Output: {self.noise_baseline:.6f} (error)")
            return self.noise_baseline
    
    def evaluate_detection(self, result):
        """Evaluación según documentación oficial CREIME_RT"""
        if result:
            if result['confidence'] > self.detection_threshold:
                self.detection_buffer.append(True)
                logging.debug(f"Ventana detectada como evento: {result['confidence']:.6f} > {self.detection_threshold}")
            else:
                self.detection_buffer.append(False)
                logging.debug(f"Ventana detectada como ruido: {result['confidence']:.6f} <= {self.detection_threshold}")
            
            if len(self.detection_buffer) >= self.consecutive_windows:
                recent_detections = list(self.detection_buffer)[-self.consecutive_windows:]
                consecutive_count = sum(recent_detections)
                
                if consecutive_count >= self.consecutive_windows:
                    logging.info(f"TRIGGER: {consecutive_count}/{self.consecutive_windows} ventanas consecutivas detectadas")
                    return {
                        'type': 'event_confirmed',
                        'consecutive_detections': consecutive_count,
                        'is_seismic': self._is_seismic_event(result)
                    }
        else:
            self.detection_buffer.append(False)
        
        return False
    
    def _is_seismic_event(self, result):
        """Determina si es evento sísmico significativo"""
        return (result['confidence'] > 0.0 and 
                result['magnitude'] is not None and 
                result['magnitude'] >= self.magnitude_threshold)
    
    def trigger_alert(self, detection_result, detection_info):
        """Activa alerta según protocolo oficial CREIME_RT"""
        self.detection_count += 1
        self.last_detection_time = detection_result['timestamp']
        
        # Calcular tiempo real del evento en el MiniSEED
        if self.miniseed_start_time and self.simulation_start_time:
            elapsed_simulation = time.time() - self.simulation_start_time
            miniseed_event_time = self.miniseed_start_time + elapsed_simulation
        else:
            miniseed_event_time = detection_result['timestamp']
        
        # Registrar primera detección
        if self.first_detection_time is None:
            self.first_detection_time = miniseed_event_time
        
        if detection_info['is_seismic']:
            # Registrar primera confirmación de sismo
            if self.first_confirmation_time is None:
                self.first_confirmation_time = miniseed_event_time
            
            alert_message = (
                f"🚨 SIMULADOR: SISMO CONFIRMADO 🚨\n"
                f"Salida CREIME_RT: {detection_result['confidence']:.6f}\n"
                f"Magnitud: {detection_result['magnitude']:.1f}\n"
                f"Ventanas consecutivas: {detection_info['consecutive_detections']}/{self.consecutive_windows}\n"
                f"Ventana: {detection_result['processing_id']}\n"
                f"Latencia: {detection_result['processing_time']:.3f}s"
            )
            logging.critical(alert_message)
            
            # Registrar evento detectado
            self.detected_events.append({
                'type': 'seismic',
                'miniseed_time': miniseed_event_time,
                'simulation_time': detection_result['timestamp'],
                'confidence': detection_result['confidence'],
                'magnitude': detection_result['magnitude'],
                'processing_id': detection_result['processing_id']
            })
            
            self.save_event_data(detection_result)
        else:
            alert_message = (
                f"⚠️ SIMULADOR: EVENTO DETECTADO ⚠️\n"
                f"Salida CREIME_RT: {detection_result['confidence']:.6f}\n"
                f"Magnitud: {detection_result['magnitude']:.1f if detection_result['magnitude'] else 'N/A'}\n"
                f"Ventanas consecutivas: {detection_info['consecutive_detections']}/{self.consecutive_windows}\n"
                f"Ventana: {detection_result['processing_id']}\n"
                f"Latencia: {detection_result['processing_time']:.3f}s"
            )
            logging.warning(alert_message)
            
            # Registrar evento detectado
            self.detected_events.append({
                'type': 'event',
                'miniseed_time': miniseed_event_time,
                'simulation_time': detection_result['timestamp'],
                'confidence': detection_result['confidence'],
                'magnitude': detection_result['magnitude'],
                'processing_id': detection_result['processing_id']
            })
    
    def save_event_data(self, detection_result):
        """Guarda datos del evento detectado en simulador"""
        try:
            events_dir = "events_simulator"
            if not os.path.exists(events_dir):
                os.makedirs(events_dir)
            
            event_id = str(uuid.uuid4())[:8]
            timestamp_str = detection_result['timestamp'].strftime('%Y%m%d_%H%M%S')
            
            json_data = {
                "station_id": self.station_id,
                "event_id": event_id,
                "timestamp": detection_result['timestamp'].isoformat(),
                "confidence": detection_result['confidence'],
                "magnitude": detection_result['magnitude'],
                "source_file": self.miniseed_file,
                "playback_speed": self.playback_speed
            }
            
            json_filename = os.path.join(events_dir, f"sim_event_{timestamp_str}.json")
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            logging.info(f"Evento simulador guardado: {json_filename}")
            
        except Exception as e:
            logging.error(f"Error guardando evento simulador: {e}")
    
    def processing_loop(self):
        """Bucle de procesamiento (igual que el original)"""
        self.last_processing_time = time.time()
        
        while self.running:
            try:
                if self.buffer.wait_for_new_data(timeout=0.3):
                    self.buffer.reset_data_event()
                    
                    result = self.ultra_fast_processing()
                    
                    if result:
                        # Capturar valor y timestamp para estadísticas
                        self.creime_values.append(result['confidence'])
                        
                        # Capturar timestamp MiniSEED correspondiente
                        if self.miniseed_start_time and self.simulation_start_time:
                            elapsed_simulation = time.time() - self.simulation_start_time
                            miniseed_time = self.miniseed_start_time + elapsed_simulation
                            self.creime_timestamps.append(miniseed_time)
                        else:
                            self.creime_timestamps.append(result['timestamp'])
                        
                        detection_info = self.evaluate_detection(result)
                        if detection_info:
                            self.trigger_alert(result, detection_info)
                
                time.sleep(0.05)
                
            except Exception as e:
                logging.error(f"Error en bucle procesamiento simulador: {e}")
                time.sleep(0.1)
    
    def start_simulation(self):
        """Inicia el simulador completo"""
        if not self.load_miniseed_data():
            return False
        
        self.start_time = time.time()
        
        logging.info("INICIANDO SIMULADOR CREIME_RT")
        logging.info(f" Archivo: {self.miniseed_file}")
        logging.info(f" Velocidad: {self.playback_speed}x")
        logging.info(f" Umbral Detección: {self.detection_threshold}")
        logging.info(f" Ventanas Consecutivas: {self.consecutive_windows}")
        if self.miniseed_start_time:
            logging.info(f" Tiempo inicio MiniSEED: {self.miniseed_start_time}")
        
        self.running = True
        
        # Hilo de simulación de datos
        self.data_thread = threading.Thread(
            target=self.simulate_data_stream,
            name="MiniSeedSimulator",
            daemon=True
        )
        self.data_thread.start()
        
        # Hilo de procesamiento
        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            name="SimulatorProcessor", 
            daemon=True
        )
        self.processing_thread.start()
        
        # Iniciar visualizador
        self.visualizer.start_visualization()
        
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
                logging.info(f"Inicializando buffer simulador: {current_samples}/1000 muestras ({elapsed:.1f}s)")
        
        if buffer_ready:
            logging.info("SIMULADOR OPERATIVO - Procesando archivo MiniSEED")
        else:
            logging.warning("Simulador operativo con buffer parcial")
        
        return True
    
    def plot_creime_output_timeline(self):
        """Genera gráfico de raw output CREIME_RT vs tiempo MiniSEED"""
        if not self.creime_values or not VISUALIZATION_ENABLED:
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Convertir timestamps a datetime si es necesario
            if self.creime_timestamps:
                times = []
                for ts in self.creime_timestamps:
                    if hasattr(ts, 'datetime'):
                        times.append(ts.datetime)
                    elif isinstance(ts, datetime):
                        times.append(ts)
                    else:
                        times.append(datetime.fromisoformat(str(ts).replace('Z', '+00:00')))
            else:
                # Fallback si no hay timestamps
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
            ax.set_xlabel('Tiempo MiniSEED', fontsize=12)
            ax.set_ylabel('CREIME_RT Raw Output', fontsize=12)
            ax.set_title('Evolución Temporal CREIME_RT - Simulador', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Formatear eje X si son timestamps reales
            if self.creime_timestamps and hasattr(self.creime_timestamps[0], 'datetime'):
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
    
    def stop_simulation(self):
        """Detiene el simulador"""
        self.running = False
        
        self.processing_pipeline.stop_workers()
        self.visualizer.stop_visualization()
        
        if self.start_time:
            run_time = time.time() - self.start_time
            processing_rate = self.processing_count / run_time if run_time > 0 else 0
            
            logging.info(f"\n{'='*60}")
            logging.info(f"REPORTE FINAL - SIMULADOR CREIME_RT")
            logging.info(f"{'='*60}")
            logging.info(f"Archivo MiniSEED:")
            logging.info(f"  Duración: {self.miniseed_duration:.1f} segundos")
            if self.miniseed_start_time:
                logging.info(f"  Inicio: {self.miniseed_start_time}")
            logging.info(f"")
            logging.info(f"Rendimiento del Sistema:")
            logging.info(f"  Tiempo procesamiento: {run_time:.1f} segundos")
            logging.info(f"  Velocidad: {self.miniseed_duration/run_time:.2f}x tiempo real")
            logging.info(f"  Paquetes procesados: {self.packet_count}")
            logging.info(f"  Ventanas CREIME_RT: {self.processing_count}")
            logging.info(f"  Tasa procesamiento: {processing_rate:.2f} ventanas/segundo")
            logging.info(f"")
            logging.info(f"Eventos Detectados: {len(self.detected_events)}")
            
            # Tiempos de detección en MiniSEED
            if self.first_detection_time:
                logging.info(f"Primera Detección (MiniSEED): {self.first_detection_time}")
            if self.first_confirmation_time:
                logging.info(f"Primera Confirmación (MiniSEED): {self.first_confirmation_time}")
            
            # Estadísticas de valores CREIME_RT para diagnóstico
            if hasattr(self, 'creime_values') and self.creime_values:
                min_val = min(self.creime_values)
                max_val = max(self.creime_values)
                mean_val = sum(self.creime_values) / len(self.creime_values)
                above_threshold = sum(1 for v in self.creime_values if v > self.detection_threshold)
                logging.info(f"")
                logging.info(f"Estadísticas CREIME_RT:")
                logging.info(f"  Valores mínimo/máximo: {min_val:.6f} / {max_val:.6f}")
                logging.info(f"  Valor promedio: {mean_val:.6f}")
                logging.info(f"  Ventanas > umbral ({self.detection_threshold}): {above_threshold}/{len(self.creime_values)}")
                logging.info(f"  Porcentaje activación: {(above_threshold/len(self.creime_values)*100):.2f}%")
            
            if self.detected_events:
                logging.info(f"Detalle de Eventos:")
                for i, event in enumerate(self.detected_events, 1):
                    event_type = "SÍSMICO" if event['type'] == 'seismic' else "EVENTO"
                    logging.info(f"  {i}. {event_type}:")
                    logging.info(f"     Tiempo MiniSEED: {event['miniseed_time']}")
                    logging.info(f"     Confianza: {event['confidence']:.6f}")
                    logging.info(f"     Magnitud: {event['magnitude']:.1f if event['magnitude'] else 'N/A'}")
                    logging.info(f"     Ventana: {event['processing_id']}")
            else:
                logging.info(f"  No se detectaron eventos sísmicos")
                logging.info(f"  Verifique que el archivo contenga actividad sísmica")
                logging.info(f"  o ajuste los umbrales de detección")
            
            logging.info(f"{'='*60}")
            
            # Generar gráfico de raw output
            self.plot_creime_output_timeline()

def main():
    """Función principal del simulador"""
    import argparse
    
    # OPCIÓN 1: Hardcodear archivo (descomenta y modifica la siguiente línea)
    # MINISEED_FILE = "mi_archivo.mseed"  # <-- AQUÍ PONES TU ARCHIVO
    
    parser = argparse.ArgumentParser(description='Simulador CREIME_RT con archivos MiniSEED')
    parser.add_argument('miniseed_file', nargs='?', help='Archivo MiniSEED de entrada (mínimo 15 segundos)')
    parser.add_argument('--model_path', default='../saipy/saved_models/', help='Ruta del modelo CREIME_RT')
    parser.add_argument('--speed', type=float, default=1.0, help='Velocidad de reproducción (1.0 = tiempo real)')
    
    args = parser.parse_args()
    
    # Si hay archivo hardcodeado, usarlo
    # if 'MINISEED_FILE' in locals():
    #     args.miniseed_file = MINISEED_FILE
    
    # Validar archivo
    if not args.miniseed_file:
        logging.error("Debe especificar un archivo MiniSEED")
        logging.info("Uso: python CREIME_RT_Simulator.py archivo.mseed")
        return
        
    if not os.path.exists(args.miniseed_file):
        logging.error(f"Archivo no encontrado: {args.miniseed_file}")
        return
    
    # Crear directorios
    for directory in ["logs", "events_simulator"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Verificar SAIPy
    try:
        import saipy
        logging.info("SAIPy disponible para simulador")
    except ImportError:
        logging.error("SAIPy no disponible")
        sys.exit(1)
    
    # Crear simulador
    simulator = MiniSeedSimulator(
        miniseed_file=args.miniseed_file,
        model_path=args.model_path,
        playback_speed=args.speed
    )
    
    try:
        logging.info(f"Iniciando simulador CREIME_RT")
        logging.info(f"LONGITUD MÍNIMA REQUERIDA: 15 segundos (1500 muestras por componente)")
        
        if simulator.start_simulation():
            if VISUALIZATION_ENABLED:
                plt.show()
            else:
                while simulator.running:
                    time.sleep(5)
        else:
            logging.error("Fallo en inicio del simulador")
            
    except KeyboardInterrupt:
        logging.info("Simulador detenido por usuario")
    except Exception as e:
        logging.error(f"Error crítico en simulador: {e}")
    finally:
        simulator.stop_simulation()

if __name__ == "__main__":
    logging.info("Ejecutando Simulador CREIME_RT")
    main()
