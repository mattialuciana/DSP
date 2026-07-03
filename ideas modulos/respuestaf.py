import numpy as np
import soundfile as sf
from modulos.senales_temporales import *
from modulos.graficos import *

def respuesta_f(entrada, salida, fs):
    """
    Calcula la respuesta en frecuencia de un sistema dado su señal de entrada y salida.
    
    Parámetros
    ----------
    entrada : NumPy array
        Señal de entrada al sistema.
    salida : NumPy array
        Señal de salida del sistema.
    fs : int
        Frecuencia de muestreo en Hz.
        
    Retorna
    -------
    freqs : NumPy array
        Frecuencias correspondientes a la respuesta en frecuencia.
    H : NumPy array
        Respuesta en frecuencia del sistema (magnitud y fase).
    """
    # Convertir a float64 para evitar problemas con FFT
    entrada = np.asarray(entrada, dtype=np.float64)
    salida = np.asarray(salida, dtype=np.float64)
    
    # Aseguramos que ambas señales tengan la misma longitud con zero padding
    max_len = max(len(entrada), len(salida))
    entrada_padded = np.pad(entrada, (0, max_len - len(entrada)), mode='constant')
    salida_padded = np.pad(salida, (0, max_len - len(salida)), mode='constant')
    
    # Calculamos la FFT de ambas señales
    N = len(entrada_padded)
    E = np.fft.fft(entrada_padded)
    S = np.fft.fft(salida_padded)
    
    # Evitamos divisiones por cero
    E_magnitude = np.abs(E)
    E_magnitude[E_magnitude == 0] = 1e-10  # Pequeño valor para evitar división por cero
    
    # Calculamos la respuesta en frecuencia H(f) = S(f) / E(f)
    H = S / E
    
    magnitud_H = np.abs(H)
    fase_H = np.angle(H)
    
    # Calculamos las frecuencias correspondientes a cada bin de la FFT
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    return freqs, magnitud_H, fase_H

"""
tono_1 = generar_tono_puro(1, 1, 1000, 50)
tono_2 = generar_tono_puro(2, 1, 1000, 50)

freq, H, faseH = respuesta_f(tono_1, tono_2, 1000)
graficar_analisis(freq, H, faseH)
"""