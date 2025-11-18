#!/usr/bin/env python3
"""
SISTEMA DE ALERTA SÍSMICA TEMPRANA - LATENCIA MÍNIMA
Sistema optimizado para Jetson Orin Nano con operación 24/7
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
import glob
import sys
import subprocess

# ===== CONFIGURACIÓN GPU SEGURA PARA JETSON ORIN NANO =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configuración de logging optimizada
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "seismic_system.log")

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
DISPLAY_SECONDS = 30
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
    """
    Sistema de visualización en tiempo real CON MARCADORES CREIME_RT
    """
    
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
        self.creime_window_start = None
        self.creime_window_end = None
        self.last_processing_time = 0
        
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
        """Configuración con marcadores CREIME_RT"""
        if not self.visualization_enabled:
            return
            
        try:
            # Configuración profesional de matplotlib
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
            
            # Configuración de la visualización - 3 subgráficas
            self.fig, self.axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120)
            self.ax1, self.ax2, self.ax3 = self.axes
            
            # Título principal
            self.fig.suptitle('SKYALERT AI-SEISMIC STATION ORIN_NANO', 
                             fontsize=16, fontweight='bold')
            
            # Líneas de gráfico
            self.line_enz, = self.ax1.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENZ')
            self.line_ene, = self.ax2.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENE')
            self.line_enn, = self.ax3.plot([], [], color=COLOR_TEAL, linewidth=1.0, label='ENN')
            
            # Configurar ejes
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
            
            # Texto para mostrar información del sistema
            self.info_text = self.ax3.text(0.02, 0.95, '', transform=self.ax3.transAxes, 
                                          fontsize=10, verticalalignment='top', color=COLOR_ORANGE,
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Inicializar marcadores CREIME_RT
            self.setup_creime_markers()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            logging.info("Visualizador configurado correctamente con marcadores CREIME_RT")
            
        except Exception as e:
            logging.error(f"Error configurando visualización: {e}")
            self.visualization_enabled = False
    
    def setup_creime_markers(self):
        """Configura los marcadores visuales para la ventana CREIME_RT"""
        try:
            # Crear líneas verticales punteadas para marcar la ventana CREIME_RT
            for ax in [self.ax1, self.ax2, self.ax3]:
                # Línea de inicio de ventana (2 segundos atrás)
                start_line = ax.axvline(x=0, color=COLOR_RED, linestyle='--', 
                                      linewidth=2, alpha=0.7, label='Inicio Ventana CREIME_RT')
                # Línea de fin de ventana (tiempo actual de procesamiento)
                end_line = ax.axvline(x=0, color=COLOR_RED, linestyle='-', 
                                    linewidth=2, alpha=0.7, label='Fin Ventana CREIME_RT')
                
                # Área sombreada entre las líneas
                fill = ax.axvspan(0, 0, color=COLOR_RED, alpha=0.1, label='Ventana CREIME_RT (2s)')
                
                self.creime_markers.append({
                    'start_line': start_line,
                    'end_line': end_line,
                    'fill': fill
                })
            
            # Agregar leyenda para los marcadores (solo en el primer subplot)
            self.ax1.legend(loc='upper left', fontsize=9)
            
        except Exception as e:
            logging.error(f"Error configurando marcadores CREIME_RT: {e}")
    
    def update_creime_markers(self, current_time_sec, rel_times):
        """Actualiza la posición de los marcadores CREIME_RT"""
        if not hasattr(self.detector, 'last_processing_time'):
            return
            
        processing_time = self.detector.last_processing_time
        
        # Solo actualizar si tenemos un tiempo de procesamiento válido
        if processing_time > 0:
            # Calcular posición relativa del fin de la ventana (último procesamiento)
            end_pos = current_time_sec - processing_time
            
            # La ventana CREIME_RT es de 2 segundos, calcular inicio
            start_pos = end_pos + 2  # 2 segundos de ventana
            
            # Verificar que las posiciones estén dentro del rango visible
            if start_pos > DISPLAY_SECONDS:
                start_pos = DISPLAY_SECONDS
            if end_pos < 0:
                end_pos = 0
            
            # Actualizar marcadores en todos los subplots
            for markers in self.creime_markers:
                markers['start_line'].set_xdata([start_pos, start_pos])
                markers['end_line'].set_xdata([end_pos, end_pos])
                
                # Actualizar área sombreada
                markers['fill'].remove()
                ax = markers['start_line'].axes
                new_fill = ax.axvspan(start_pos, end_pos, color=COLOR_RED, alpha=0.1)
                markers['fill'] = new_fill
    
    def calculate_dynamic_ylimits(self, data, component_name):
        """Algoritmo para escalado dinámico"""
        if len(data) == 0:
            return -1, 1
        
        current_max = np.max(np.abs(data))
        self.max_values_history[component_name].append(current_max)
        
        if len(self.max_values_history[component_name]) > 0:
            historical_max = np.percentile(list(self.max_values_history[component_name]), 90)
            target_max = max(current_max, historical_max)
        else:
            target_max = current_max
        
        margin = 1.25
        limit = target_max * margin
        min_limit = 0.1
        
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
        """Actualización de gráficos CON MARCADORES CREIME_RT"""
        if not self.visualization_enabled or not self.fig:
            artists = [self.line_enz, self.line_ene, self.line_enn, self.info_text]
            for markers in self.creime_markers:
                artists.extend([markers['start_line'], markers['end_line'], markers['fill']])
            return artists
        
        with self.lock:
            times_copy = np.array(self.times)
            enz_copy = np.array(self.data_enz)
            ene_copy = np.array(self.data_ene)
            enn_copy = np.array(self.data_enn)
        
        min_len = min(len(times_copy), len(enz_copy), len(ene_copy), len(enn_copy))
        if min_len < 10:
            artists = [self.line_enz, self.line_ene, self.line_enn, self.info_text]
            for markers in self.creime_markers:
                artists.extend([markers['start_line'], markers['end_line'], markers['fill']])
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
        
        self.update_creime_markers(current_time_sec, rel_times)
        
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
        for markers in self.creime_markers:
            artists.extend([markers['start_line'], markers['end_line'], markers['fill']])
        
        return artists
    
    def start_visualization(self):
        """Inicia visualización"""
        if not self.visualization_enabled:
            return
            
        try:
            self.running = True
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=150, blit=True, cache_frame_data=False
            )
            logging.info("Visualizador iniciado con marcadores CREIME_RT (2s)")
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
    """
    Buffer ultra-rápido para latencia mínima (2 segundos)
    """
    
    def __init__(self, window_size=100, sampling_rate=100, update_interval=0.5):
        # VENTANA MÍNIMA: 2 segundos (200 muestras)
        self.window_size = int(window_size)
        self.sampling_rate = int(sampling_rate)
        self.update_interval = update_interval
        self.update_samples = int(update_interval * sampling_rate)
        
        # Buffer muy pequeño para máxima velocidad
        total_size = self.window_size + (1 * self.update_samples)  # 400 muestras máximo
        self.buffers = {
            'ENZ': deque(maxlen=total_size),
            'ENE': deque(maxlen=total_size),
            'ENN': deque(maxlen=total_size)
        }
        
        self.lock = threading.Lock()
        self.window_count = 0
        self.last_window_time = 0
        
        # MODIFICACIÓN: Listo con solo 0.5 segundos de datos
        self.ready = False
        self.min_ready_samples = int(0.5 * sampling_rate)  # 50 muestras mínimas
        
        # Evento para sincronización ultra-rápida
        self.new_data_event = threading.Event()
        
        # Estadísticas
        self.performance_stats = {
            'windows_generated': 0,
            'data_points_received': 0,
            'last_update': time.time()
        }
    
    def add_data(self, component, data, timestamp):
        """Añade datos y activa evento inmediatamente"""
        with self.lock:
            if component in self.buffers:
                self.buffers[component].extend(data)
                self.performance_stats['data_points_received'] += len(data)
                
                # LISTO CON SOLO 0.5 SEGUNDOS DE DATOS
                min_samples = min(len(buf) for buf in self.buffers.values())
                self.ready = min_samples >= self.min_ready_samples
                
                # ACTIVAR EVENTO INMEDIATAMENTE
                if data:
                    self.new_data_event.set()
    
    def wait_for_new_data(self, timeout=0.5):
        """Espera ultra-rápida por nuevos datos"""
        return self.new_data_event.wait(timeout)
    
    def reset_data_event(self):
        """Resetea el evento de nuevos datos"""
        self.new_data_event.clear()
    
    def get_latest_window(self):
        """Genera ventana deslizante ultra-rápida"""
        with self.lock:
            current_time = time.time()
            
            # Procesamiento más frecuente
            if current_time - self.last_window_time < self.update_interval:
                return None
                
            if not self.ready:
                return None
            
            # Extraer ventana actual (últimos 200 puntos)
            window_data = []
            for component in ['ENZ', 'ENE', 'ENN']:
                component_data = list(self.buffers[component])
                
                # RELLENAR RÁPIDAMENTE SI NO HAY SUFICIENTES DATOS
                if len(component_data) < self.window_size:
                    padding_needed = self.window_size - len(component_data)
                    padding = [0.0] * padding_needed
                    component_data = padding + component_data
                else:
                    component_data = component_data[-self.window_size:]
                
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
    """Filtro optimizado para bajo consumo"""
    
    def __init__(self, fs=100, hp_cutoff=0.09, lp_cutoff=45):
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
    """
    Pipeline de procesamiento ultra-rápido
    """
    
    def __init__(self, model_path, num_workers=1):
        self.model_path = model_path
        self.num_workers = num_workers
        self.processing_queue = queue.Queue(maxsize=2)  # Cola muy pequeña
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
                name=f"CREIME_Worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Iniciados {self.num_workers} workers")
    
    def processing_worker(self):
        """Worker ultra-rápido para CREIME_RT"""
        try:
            from saipy.models.creime import CREIME_RT
            model = CREIME_RT(self.model_path)
            logging.info("Worker de CREIME_RT inicializado")
        except Exception as e:
            logging.error(f"Worker no pudo cargar modelo: {e}")
            return
        
        while self.running:
            try:
                window_data, processing_id = self.processing_queue.get(timeout=1.0)
                
                if window_data is None:
                    break
                
                start_time = time.time()
                
                # Procesamiento directo
                # amazonq-ignore-next-line
                y_pred, predictions = model.predict(window_data)
                result = predictions[0]
                
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
                logging.error(f"Error en worker: {e}")
                try:
                    self.processing_queue.task_done()
                except:
                    pass
                continue
    
    def submit_window(self, window_data, processing_id):
        """Envía ventana para procesamiento inmediato"""
        try:
            self.processing_queue.put((window_data, processing_id), timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.5):
        """Obtiene resultado rápidamente"""
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

class TwoSecondLatencyDetector:
    """
    Sistema principal de detección con latencia MÍNIMA (2 segundos)
    """
    
    def __init__(self, model_path, host='localhost', port=30000, sampling_rate=100):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        
        # CONFIGURACIÓN ULTRA-RÁPIDA - VENTANA MÍNIMA
        self.window_size = 2 * sampling_rate  # 200 muestras - 2 SEGUNDOS
        # PARÁMETROS OPTIMIZADOS PARA VELOCIDAD MÁXIMA
        self.latency_target = 0.1  # 300 ms entre procesamientos
        self.confidence_threshold = 0.95  # Menor umbral para mayor sensibilidad
        self.consecutive_windows = 1  # Una sola detección
        
        # Componentes ultra-rápidos
        self.buffer = UltraFastBuffer(
            window_size=self.window_size,
            sampling_rate=sampling_rate,
            update_interval=self.latency_target
        )
        
        self.hybrid_filter = OptimizedHybridFilter(fs=sampling_rate)
        self.processing_pipeline = UltraFastProcessingPipeline(model_path, num_workers=1)
        
        # Visualizador
        self.visualizer = RealTimeVisualizer(self)
        
        # Estado del sistema
        self.socket = None
        self.running = False
        self.data_buffer = b''
        
        # Estadísticas
        self.detection_count = 0
        self.last_detection_time = None
        self.packet_count = 0
        self.start_time = None
        self.processing_count = 0
        
        # Control de procesamiento
        self.last_processing_time = 0
        self.detection_buffer = deque(maxlen=3)
        
        # Hilos
        self.receiver_thread = None
        self.processing_thread = None
        
        logging.info("=== SISTEMA CONFIGURADO ===")
        logging.info(f"VENTANA MÍNIMA: {self.window_size} muestras ({self.window_size/sampling_rate} segundos)")
        logging.info(f"LATENCIA OBJETIVO: {self.latency_target} segundos")
        logging.info(f"CONFIANZA: {self.confidence_threshold}")
    
    def connect_to_observer(self):
        """Conexión rápida con AnyShake Observer"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(3.0)
                self.socket.connect((self.host, self.port))
                logging.info(f" Conexión establecida: {self.host}:{self.port}")
                return True
                
            except Exception as e:
                logging.warning(f"Intento {attempt + 1}/{max_retries} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error("No se pudo establecer conexión")
                    return False
    
    def parse_observer_packet(self, packet):
        """Parser optimizado para velocidad máxima"""
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
                data_values = [x * CONVERSION_FACTOR for x in data_values]
            
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
        """Calcula confianza ultra-rápida"""
        if result_data['result'][0] == 1:
            return 0.75  # Confianza fija para velocidad
        return 0.0
    
    def evaluate_detection(self, result):
        """Evaluación ultra-rápida de detección"""
        if result and result['detection'] == 1:
            if result['confidence'] >= self.confidence_threshold:
                self.detection_buffer.append(True)
                
                if len(self.detection_buffer) >= self.consecutive_windows:
                    return True
            else:
                self.detection_buffer.append(False)
        else:
            self.detection_buffer.append(False)
        
        return False
    
    def trigger_alert(self, detection_result):
        """Activa alerta sísmica ULTRA-RÁPIDA"""
        # UMBRAL MUY BAJO PARA PRUEBAS - AJUSTAR SEGÚN NECESIDAD
        if detection_result['magnitude'] is None or detection_result['magnitude'] < 0.6:
            logging.info(f"Evento descartado - Magnitud: {detection_result['magnitude']}")
            return
        
        self.detection_count += 1
        self.last_detection_time = detection_result['timestamp']
        
        alert_message = (
            f"🚨 ALERTA: SISMO CONFIRMADO 🚨 - "
            f"Confianza: {detection_result['confidence']:.3f} | "
            f"Magnitud: {detection_result['magnitude']:.1f} | "
            f"Ventana: {detection_result['processing_id']}"
            f"Latencia: {detection_result['processing_time']:.3f}s | "
        )
        
        logging.critical(alert_message)
        self.save_event_data(detection_result)
    
    def save_event_data(self, detection_result):
        """Guarda datos del evento rápidamente"""
        try:
            events_dir = "events"
            if not os.path.exists(events_dir):
                os.makedirs(events_dir)
                
            filename = os.path.join(events_dir, f"event_ULTRAFAST_{detection_result['timestamp'].strftime('%Y%m%d_%H%M%S')}.npz")
            np.savez_compressed(
                filename,
                timestamp=detection_result['timestamp'],
                magnitude=detection_result['magnitude'],
                confidence=detection_result['confidence'],
                processing_time=detection_result['processing_time'],
                processing_id=detection_result['processing_id'],
                waveform_data=detection_result['window_data'],
                detection_count=self.detection_count
            )
            logging.info(f"Evento ULTRA-RÁPIDO guardado: {filename}")
        except Exception as e:
            logging.error(f"Error guardando evento: {e}")
    
    def processing_loop(self):
        """Bucle de procesamiento ULTRA-RÁPIDO"""
        self.last_processing_time = time.time()
        
        while self.running:
            try:
                # ESPERA ACTIVA ULTRA-RÁPIDA
                if self.buffer.wait_for_new_data(timeout=0.3):
                    self.buffer.reset_data_event()
                    
                    result = self.ultra_fast_processing()
                    
                    if result:
                        status = " ANOMALÍA_DETECTADA" if result['detection'] == 1 else " NO_DETECTADO"
                        mag_display = f"{result['magnitude']:.1f}" if result['magnitude'] is not None else "N/A"
                        
                        logging.info(
                            f"Procesado {result['processing_id']}: {status} | "
                            f"Mag: {mag_display} | Conf: {result['confidence']:.3f} | "
                            f"Tiempo: {result['processing_time']:.3f}s"
                        )
                        
                        if self.evaluate_detection(result):
                            self.trigger_alert(result)
                
                time.sleep(0.05)  # Sleep mínimo para reducir CPU
                
            except Exception as e:
                logging.error(f"Error en bucle ULTRA-RÁPIDO: {e}")
                time.sleep(0.1)
    
    def receive_data_loop(self):
        """Bucle de recepción ULTRA-RÁPIDO"""
        self.running = True
        self.data_buffer = b''
        
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    logging.warning("Conexión cerrada - Reconectando...")
                    if not self.connect_to_observer():
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
    
    def start_system(self):
        """Inicia el sistema ULTRA-RÁPIDO"""
        try:
            os.nice(10)
        except:
            pass
        
        if not self.connect_to_observer():
            return False
        
        self.start_time = time.time()
        
        logging.info("INICIANDO SISTEMA")
        logging.info(f" Ventana: {self.window_size} muestras")
        logging.info(f" Latencia: {self.latency_target}s")
        logging.info(f" Confianza: {self.confidence_threshold}")
        
        # Hilo de recepción
        self.receiver_thread = threading.Thread(
            target=self.receive_data_loop,
            name="UltraFastReceiver",
            daemon=True
        )
        self.receiver_thread.start()
        
        # Hilo de procesamiento
        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            name="UltraFastProcessor", 
            daemon=True
        )
        self.processing_thread.start()
        
        # Iniciar visualizador
        self.visualizer.start_visualization()
        
        # INICIALIZACIÓN ULTRA-RÁPIDA - SOLO 0.5 SEGUNDOS DE ESPERA
        buffer_ready = False
        startup_time = time.time()
        while not buffer_ready and self.running and (time.time() - startup_time < 10):
            time.sleep(0.1)
            status = self.buffer.get_buffer_status()
            current_samples = status['ENZ']['samples']
            buffer_ready = current_samples >= 50  # 0.5 segundos
            
            if not buffer_ready and (time.time() - startup_time > 1):
                elapsed = time.time() - startup_time
                logging.info(f"Inicializando buffer: {current_samples}/200 muestras ({elapsed:.1f}s)")
        
        if buffer_ready:
            logging.info("SISTEMA OPERATIVO - Detección activa en 2 SEGUNDOS")
        else:
            logging.warning(" Sistema operativo con buffer parcial")
        
        return True
    
    def stop_system(self):
        """Detiene el sistema"""
        self.running = False
        
        self.processing_pipeline.stop_workers()
        self.visualizer.stop_visualization()
        
        if self.socket:
            self.socket.close()
        
        if self.start_time:
            run_time = time.time() - self.start_time
            processing_rate = self.processing_count / run_time if run_time > 0 else 0
            
            logging.info(
                f"\n=== REPORTE FINAL ==="
                f"Tiempo total: {run_time:.1f}s"
                f"Paquetes: {self.packet_count}"
                f"Procesamientos: {self.processing_count}"
                f"Tasa: {processing_rate:.2f} ventanas/segundo"
                f"Detecciones: {self.detection_count}"
                f"Última detección: {self.last_detection_time}"
            )

def main():
    """Función principal ULTRA-RÁPIDA"""
    MODEL_PATH = "../saipy/saved_models/"
    OBSERVER_HOST = "localhost"
    OBSERVER_PORT = 30000
    
    max_restarts = 2
    restart_delay = 5
    restart_count = 0
    
    for directory in ["logs", "events", "backups"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    while restart_count < max_restarts:
        detector = TwoSecondLatencyDetector(
            model_path=MODEL_PATH,
            host=OBSERVER_HOST,
            port=OBSERVER_PORT
        )
        
        try:
            logging.info(f"Iniciando sistema (intento {restart_count + 1}/{max_restarts})")
            
            if detector.start_system():
                if VISUALIZATION_ENABLED:
                    plt.show()
                else:
                    while detector.running:
                        time.sleep(5)
            else:
                logging.error("Fallo en inicio del sistema")
                restart_count += 1
                
        except KeyboardInterrupt:
            logging.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logging.error(f"Error crítico: {e}")
            restart_count += 1
            
            if restart_count < max_restarts:
                logging.info(f"Reiniciando en {restart_delay} segundos...")
                time.sleep(restart_delay)
            else:
                logging.error("Máximos reinicios alcanzados")
                break
        finally:
            detector.stop_system()

if __name__ == "__main__":
    if hasattr(os, 'uname') and 'aarch64' in os.uname().machine:
        logging.info("Ejecutando en Jetson Orin Nano")
    else:
        logging.warning("Plataforma no optimizada - rendimiento puede variar")
    
    try:
        import saipy
        logging.info("SAIPy disponible")
    except ImportError:
        logging.error("SAIPy no disponible")
        sys.exit(1)
    
    main()
