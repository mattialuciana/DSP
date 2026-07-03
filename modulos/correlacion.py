import scipy.signal as signal
from modulos import graficar_analisis, analisis_filtros
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def calcular_coherencia(entrada, salida, fs=1.0, nperseg=256):
    """
    Calcula la coherencia cuadrática entre dos señales usando la fórmula analítica.
    
    Parámetros:
    entrada : array_like - Señal de entrada.
    salida : array_like - Señal de salida.
    fs : float - Frecuencia de muestreo.
    nperseg : int - Longitud de cada segmento para el método de Welch.
    
    Retorna:
    f : ndarray - Array de frecuencias muestrales.
    coherencia : ndarray - Valores de la coherencia cuadrática para cada frecuencia.
    """
    
    # Autocorr entrada
    f, Gxx = signal.welch(entrada, fs=fs, nperseg=nperseg)
    
    # Autocorr salida
    _, Gyy = signal.welch(salida, fs=fs, nperseg=nperseg)
    
    # Corr cruzado
    _, Gxy = signal.csd(entrada, salida, fs=fs, nperseg=nperseg)
    
    numerador = np.abs(Gxy)**2
    denominador = Gxx * Gyy
    
    # Precaución numérica: evitamos dividir por cero si Gxx o Gyy son 0 en alguna frecuencia
    # coherencia = np.zeros_like(numerador)

    # Manejo de ceros
    #epsilon = 1e-12 
    #denominador = denominador + epsilon

    coherencia = numerador / denominador

    H = Gxy / Gxx
    mod = np.abs(H)
    fase = np.angle(H)
    
    return f, coherencia, mod, fase

def graficar_coherencia(f, coherencia):
    """
    Grafica la coherencia cuadrática en función de la frecuencia.
    
    Parámetros:
    f : ndarray - Array de frecuencias en Hz.
    coherencia : ndarray - Valores de la coherencia cuadrática.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(f, coherencia, color='blue', linewidth=1.5)
    
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Coherencia cuadrática')
    
    plt.ylim(0, 1.01)
    
    plt.title('Análisis de Coherencia Cuadrática')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlim(left=0) 
    
    plt.show()

entrada, fs = sf.read(r'archivos_parte2_tp1c2026\entrada.wav') 
salida1, fs1 = sf.read(r'archivos_parte2_tp1c2026\salida1.wav')
salida2, fs2 = sf.read(r'archivos_parte2_tp1c2026\salida2.wav') 
salida3, fs3 = sf.read(r'archivos_parte2_tp1c2026\salida3.wav')

f1, coherencia1, mod1, fase1 = calcular_coherencia(entrada, salida1, fs)
f2, coherencia2, mod2, fase2 = calcular_coherencia(entrada, salida2, fs)
f3, coherencia3, mod3, fase3 = calcular_coherencia(entrada,salida3, fs)

# CHEQUEO
# f, coherencia = signal.coherence(entrada, salida1, fs=fs, nperseg=256)
graficar_coherencia(f1, coherencia1)
graficar_analisis(f1, mod1, fase1, titulo='Módulo y fase de la estimación de H')
graficar_coherencia(f2, coherencia2)
graficar_analisis(f2, mod2, fase2, titulo='Módulo y fase de la estimación de H')
graficar_coherencia(f3, coherencia3)
graficar_analisis(f3, mod3, fase3, titulo='Módulo y fase de la estimación de H')
