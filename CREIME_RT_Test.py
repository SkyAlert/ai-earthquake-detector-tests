#!/usr/bin/env python3
"""
SCRIPT SIMPLIFICADO PARA ENTENDER CREIME_RT
Análisis detallado de entrada y salida del modelo
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from scipy.signal import butter, filtfilt
import sys
import argparse

def apply_preprocessing(data, sampling_rate=100):
    """Aplica preprocesamiento según documentación CREIME_RT"""
    # 1. Filtrado pasa-banda [1.0, 45.0] Hz
    nyquist = 0.5 * sampling_rate
    low = 1.0 / nyquist
    high = 45.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    # 2. Normalización z-score
    mean_val = np.mean(filtered_data)
    std_val = np.std(filtered_data)
    if std_val > 0:
        normalized_data = (filtered_data - mean_val) / std_val
    else:
        normalized_data = filtered_data
    
    return normalized_data.astype(np.float32)

def extract_slices(miniseed_file, earthquake_time="2025-01-12T08:32:58"):
    """Extrae rebanadas de ruido y sismo según documentación CREIME_RT"""
    print("="*60)
    print("ANÁLISIS CREIME_RT - EXTRACCIÓN DE REBANADAS")
    print("="*60)
    
    # Cargar MiniSEED
    stream = read(miniseed_file)
    print(f"Archivo cargado: {miniseed_file}")
    print(f"Canales disponibles: {[tr.stats.channel for tr in stream]}")
    
    # Mapear componentes
    components = {}
    for tr in stream:
        channel = tr.stats.channel.upper()
        if 'Z' in channel:
            components['Z'] = tr
        elif 'E' in channel or '1' in channel:
            components['E'] = tr
        elif 'N' in channel or '2' in channel:
            components['N'] = tr
    
    if len(components) < 3:
        raise ValueError(f"Se requieren 3 componentes, encontrados: {len(components)}")
    
    # Información del archivo
    start_time = components['Z'].stats.starttime
    sampling_rate = components['Z'].stats.sampling_rate
    duration = len(components['Z'].data) / sampling_rate
    
    print(f"Inicio archivo: {start_time}")
    print(f"Duración: {duration:.1f} segundos")
    print(f"Frecuencia muestreo: {sampling_rate} Hz")
    
    # Tiempo del sismo
    eq_time = UTCDateTime(earthquake_time)
    eq_offset = eq_time - start_time  # segundos desde inicio
    
    print(f"Tiempo sismo: {eq_time}")
    print(f"Offset sismo: {eq_offset:.1f} segundos desde inicio")
    
    # Según documentación: ventanas de 30 segundos (3000 muestras a 100 Hz)
    window_duration = 30  # segundos
    samples_per_window = int(window_duration * sampling_rate)
    
    print(f"Ventana CREIME_RT: {window_duration}s ({samples_per_window} muestras)")
    
    # REBANADA 1: RUIDO (30s antes del sismo)
    noise_start_offset = eq_offset - 60  # 60s antes del sismo
    noise_start_sample = int(noise_start_offset * sampling_rate)
    noise_end_sample = noise_start_sample + samples_per_window
    
    # REBANADA 2: SISMO (desde inicio del sismo)
    sismo_start_sample = int(eq_offset * sampling_rate)
    sismo_end_sample = sismo_start_sample + samples_per_window
    
    print(f"\nREBANADA 1 (RUIDO):")
    print(f"  Tiempo: {noise_start_offset:.1f}s a {noise_start_offset + window_duration:.1f}s")
    print(f"  Muestras: {noise_start_sample} a {noise_end_sample}")
    
    print(f"\nREBANADA 2 (SISMO):")
    print(f"  Tiempo: {eq_offset:.1f}s a {eq_offset + window_duration:.1f}s")
    print(f"  Muestras: {sismo_start_sample} a {sismo_end_sample}")
    
    # Extraer datos
    slices = {}
    
    for slice_name, start_idx, end_idx in [("noise", noise_start_sample, noise_end_sample),
                                          ("earthquake", sismo_start_sample, sismo_end_sample)]:
        
        slice_data = []
        for comp_name in ['Z', 'E', 'N']:
            tr = components[comp_name]
            
            # Verificar límites
            if start_idx < 0 or end_idx > len(tr.data):
                print(f"ADVERTENCIA: {slice_name} fuera de límites para {comp_name}")
                # Rellenar con ceros si es necesario
                if start_idx < 0:
                    padding_start = abs(start_idx)
                    start_idx = 0
                else:
                    padding_start = 0
                
                if end_idx > len(tr.data):
                    padding_end = end_idx - len(tr.data)
                    end_idx = len(tr.data)
                else:
                    padding_end = 0
                
                raw_data = tr.data[start_idx:end_idx]
                
                # Agregar padding si es necesario
                if padding_start > 0:
                    raw_data = np.concatenate([np.zeros(padding_start), raw_data])
                if padding_end > 0:
                    raw_data = np.concatenate([raw_data, np.zeros(padding_end)])
            else:
                raw_data = tr.data[start_idx:end_idx]
            
            # Aplicar preprocesamiento
            processed_data = apply_preprocessing(raw_data, sampling_rate)
            slice_data.append(processed_data)
        
        # Formato para CREIME_RT: (3000, 3)
        slice_array = np.column_stack(slice_data)
        slices[slice_name] = {
            'data': slice_array,
            'raw_data': np.column_stack([components[comp].data[max(0, start_idx):min(len(components[comp].data), end_idx)] 
                                       for comp in ['Z', 'E', 'N']]),
            'start_time': start_time + start_idx / sampling_rate,
            'samples': slice_array.shape[0]
        }
        
        print(f"\n{slice_name.upper()} - Forma final: {slice_array.shape}")
        print(f"  Rango datos procesados: [{np.min(slice_array):.6f}, {np.max(slice_array):.6f}]")
    
    return slices, sampling_rate

def plot_data(slices, sampling_rate):
    """Grafica las rebanadas de datos"""
    print("\n" + "="*60)
    print("VISUALIZACIÓN DE DATOS PARA CREIME_RT")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Datos Procesados para CREIME_RT\n(Filtrado [1-45Hz] + Z-Score)', fontsize=16, fontweight='bold')
    
    time_axis = np.arange(3000) / sampling_rate  # 30 segundos
    
    components = ['Z (Vertical)', 'E (Este)', 'N (Norte)']
    
    for i, (slice_name, slice_info) in enumerate(slices.items()):
        data = slice_info['data']
        
        for j, comp_name in enumerate(components):
            ax = axes[i, j]
            ax.plot(time_axis, data[:, j], 'b-', linewidth=0.8)
            ax.set_title(f'{slice_name.upper()} - {comp_name}')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud Normalizada')
            ax.grid(True, alpha=0.3)
            
            # Estadísticas
            mean_val = np.mean(data[:, j])
            std_val = np.std(data[:, j])
            max_val = np.max(np.abs(data[:, j]))
            
            ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmax={max_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def analyze_creime_output(slices):
    """Analiza la salida de CREIME_RT"""
    print("\n" + "="*60)
    print("ANÁLISIS DE SALIDA CREIME_RT")
    print("="*60)
    
    try:
        from saipy.models.creime import CREIME_RT
        
        # Cargar modelo
        model_path = "../saipy/saved_models/"
        print(f"Cargando modelo desde: {model_path}")
        model = CREIME_RT(model_path)
        print("✅ Modelo CREIME_RT cargado exitosamente")
        
        for slice_name, slice_info in slices.items():
            print(f"\n{'='*40}")
            print(f"ANÁLISIS: {slice_name.upper()}")
            print(f"{'='*40}")
            
            # Preparar datos para CREIME_RT
            input_data = np.expand_dims(slice_info['data'], axis=0)  # (1, 3000, 3)
            
            print(f"Forma de entrada: {input_data.shape}")
            print(f"Tipo de datos: {input_data.dtype}")
            print(f"Rango de valores: [{np.min(input_data):.6f}, {np.max(input_data):.6f}]")
            
            # Ejecutar predicción
            print("\nEjecutando predicción CREIME_RT...")
            y_pred, predictions = model.predict(input_data)
            
            print(f"\n📊 RESULTADOS CREIME_RT:")
            print(f"   Tipo y_pred: {type(y_pred)}")
            print(f"   Forma y_pred: {y_pred.shape if hasattr(y_pred, 'shape') else 'N/A'}")
            print(f"   Tipo predictions: {type(predictions)}")
            
            # Analizar y_pred (salida principal)
            if y_pred is not None:
                print(f"\n🔍 ANÁLISIS y_pred:")
                if hasattr(y_pred, 'shape'):
                    print(f"   Dimensiones: {y_pred.shape}")
                    if len(y_pred.shape) > 1:
                        print(f"   Primeros 10 valores: {y_pred[0][:10]}")
                        print(f"   Últimos 10 valores: {y_pred[0][-10:]}")
                        print(f"   Valor mínimo: {np.min(y_pred):.6f}")
                        print(f"   Valor máximo: {np.max(y_pred):.6f}")
                        print(f"   Valor medio: {np.mean(y_pred):.6f}")
                        
                        # Análisis según documentación
                        final_value = y_pred[0][-1] if len(y_pred[0]) > 0 else y_pred[0]
                        print(f"\n📈 INTERPRETACIÓN (valor final: {final_value:.6f}):")
                        
                        if final_value <= -4.0:
                            interpretation = "🔵 RUIDO PURO"
                        elif final_value <= -0.5:
                            interpretation = "🟡 SEÑAL MIXTA (por debajo umbral)"
                        else:
                            interpretation = "🔴 EVENTO DETECTADO (por encima umbral -0.5)"
                            if final_value > 0:
                                interpretation += f" - Magnitud estimada: {final_value:.2f}"
                        
                        print(f"   {interpretation}")
                        
                        # Contar valores por encima del umbral
                        above_threshold = np.sum(y_pred[0] > -0.5)
                        total_samples = len(y_pred[0])
                        percentage = (above_threshold / total_samples) * 100
                        print(f"   Muestras > -0.5: {above_threshold}/{total_samples} ({percentage:.1f}%)")
                    else:
                        print(f"   Valor único: {y_pred[0]:.6f}")
                else:
                    print(f"   Valor: {y_pred}")
            
            # Analizar predictions
            if predictions is not None:
                print(f"\n🔍 ANÁLISIS predictions:")
                print(f"   Contenido: {predictions}")
                if len(predictions) > 0:
                    pred = predictions[0]
                    print(f"   Detección binaria: {pred[0] if len(pred) > 0 else 'N/A'}")
                    print(f"   Magnitud estimada: {pred[1] if len(pred) > 1 else 'N/A'}")
            
            print(f"\n{'='*40}")
    
    except ImportError:
        print("❌ Error: SAIPy no está disponible")
        print("   Instale SAIPy para ejecutar CREIME_RT")
    except Exception as e:
        print(f"❌ Error ejecutando CREIME_RT: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Análisis simplificado de CREIME_RT')
    parser.add_argument('miniseed_file', help='Archivo MiniSEED de entrada')
    parser.add_argument('--earthquake_time', default='2025-01-12T08:32:58', 
                       help='Tiempo del sismo (formato: YYYY-MM-DDTHH:MM:SS)')
    
    args = parser.parse_args()
    
    try:
        # 1. Extraer rebanadas
        slices, sampling_rate = extract_slices(args.miniseed_file, args.earthquake_time)
        
        # 2. Graficar datos
        plot_data(slices, sampling_rate)
        
        # 3. Analizar salida CREIME_RT
        analyze_creime_output(slices)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO")
        print("="*60)
        print("✅ Rebanadas extraídas correctamente")
        print("✅ Datos graficados")
        print("✅ Análisis CREIME_RT ejecutado")
        print("\nUse este análisis para entender el comportamiento de CREIME_RT")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()