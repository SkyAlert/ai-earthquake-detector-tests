#!/usr/bin/env python3
"""
Test simple para verificar modo tiempo real de AnyShake
"""

import socket
import time
from datetime import datetime

def enable_realtime():
    """Activa modo tiempo real"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 30000))
        s.sendall(b"AT+REALTIME=1\r\n")
        s.close()
        print("✅ Comando AT+REALTIME=1 enviado")
        return True
    except Exception as e:
        print(f"❌ Error activando tiempo real: {e}")
        return False

def test_packet_rate():
    """Prueba la velocidad de paquetes"""
    print("🔍 Activando modo tiempo real...")
    enable_realtime()
    time.sleep(2)
    
    print("📡 Conectando para recibir datos...")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 30000))
        s.settimeout(1.0)
        
        packet_count = 0
        start_time = time.time()
        last_packet_time = start_time
        data_buffer = b''
        
        print("📊 Monitoreando paquetes (Ctrl+C para parar)...")
        print("Formato: [Tiempo] Paquete #N - Intervalo: Xms - Tasa: X pkt/s")
        print("-" * 60)
        
        while True:
            try:
                data = s.recv(4096)
                if not data:
                    break
                
                data_buffer += data
                
                while b'\r' in data_buffer:
                    packet, data_buffer = data_buffer.split(b'\r', 1)
                    packet_str = packet.decode('ascii', errors='ignore').strip()
                    
                    if packet_str and packet_str.startswith('$'):
                        current_time = time.time()
                        packet_count += 1
                        
                        # Calcular intervalo entre paquetes
                        interval_ms = (current_time - last_packet_time) * 1000
                        
                        # Calcular tasa de paquetes
                        elapsed = current_time - start_time
                        rate = packet_count / elapsed if elapsed > 0 else 0
                        
                        # Mostrar cada 10 paquetes
                        if packet_count % 10 == 0:
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"[{timestamp}] Paquete #{packet_count:3d} - Intervalo: {interval_ms:6.1f}ms - Tasa: {rate:5.1f} pkt/s")
                        
                        last_packet_time = current_time
                        
                        # Análisis después de 30 paquetes
                        if packet_count == 30:
                            avg_interval = (elapsed / packet_count) * 1000
                            if avg_interval < 200:
                                print(f"🚀 MODO RÁPIDO DETECTADO - Intervalo promedio: {avg_interval:.1f}ms")
                            else:
                                print(f"🐢 Modo normal - Intervalo promedio: {avg_interval:.1f}ms")
                            
            except socket.timeout:
                continue
                
    except KeyboardInterrupt:
        print("\n⏹️ Detenido por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            s.close()
        except:
            pass
        
        # Estadísticas finales
        total_time = time.time() - start_time
        if packet_count > 0:
            avg_rate = packet_count / total_time
            avg_interval = (total_time / packet_count) * 1000
            
            print("\n📈 ESTADÍSTICAS FINALES:")
            print(f"   Paquetes recibidos: {packet_count}")
            print(f"   Tiempo total: {total_time:.1f}s")
            print(f"   Tasa promedio: {avg_rate:.1f} pkt/s")
            print(f"   Intervalo promedio: {avg_interval:.1f}ms")
            
            if avg_interval < 200:
                print("   Estado: 🚀 MODO TIEMPO REAL")
            else:
                print("   Estado: 🐢 MODO NORMAL")

if __name__ == "__main__":
    print("🧪 Test AnyShake - Verificación Modo Tiempo Real")
    print("=" * 50)
    test_packet_rate()