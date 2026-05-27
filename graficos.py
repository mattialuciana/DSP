
# GRAFICAR FUNCIONES EN EL TIEMPO. revisar los rastros de copilot

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def graficar_t(fs, señales, etiquetas=None, titulo='Señales en el Tiempo', xlim=None):
    """
    Grafica una o varias señales en función del tiempo calculando 
    la duración automáticamente y permitiendo ajustar los límites del eje X.
    
    Parámetros
    ----------
    fs : int
        Frecuencia de muestreo en Hz de la señal.
    señales : NumPy array o lista de NumPy arrays
        Datos de la o las señales generadas.
    etiquetas : str o lista de str, opcional
        Etiquetas para identificar cada señal en la leyenda.
    titulo : str, opcional
        Título del gráfico.
    xlim : tuple o list, opcional
        Límites para el eje X, ej: (0, 0.01). Si es None, muestra el tiempo completo.
    """
    # Si se recibe una única señal, la convertimos en lista
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    if etiquetas is None:
        etiquetas = [f'Señal {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Recorremos cada señal
    for s, etiqueta in zip(señales, etiquetas):
        muestras = len(s) 
        duracion = muestras / fs 
        t = np.arange(0, duracion, 1/fs)
        
        ax.plot(t[:muestras], s, label=etiqueta)
    
    # Si el usuario definió xlim, lo aplicamos al eje X
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.legend(loc='upper right') 
    ax.grid(True)
    plt.show()
    
# GRAFICAR FUNCIONES EN FRECUENCIA. revisar los rastros de copilot. cuidado con la escala logarítmica  (límites?)
def graficar_f(fs, señales, etiquetas=None, titulo='Espectro de Amplitud (Fourier)', xlim=(20, 20000)):
    """
    Calcula y grafica el espectro de magnitud de Fourier en escala logarítmica.
    
    Parámetros
    ----------
    fs : int
        Frecuencia de muestreo en Hz.
    señales : NumPy array o lista de NumPy arrays
        Datos de la o las señales de entrada.
    etiquetas : str o lista de str, opcional
        Etiquetas para identificar cada espectro en la leyenda.
    titulo : str, opcional
        Título del gráfico.
    xlim : tuple, opcional
        Límites del eje X. Por defecto (20, 20000) Hz. En escala log, debe ser > 0.
    """
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    if etiquetas is None:
        etiquetas = [f'Espectro {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))

    for s, etiqueta in zip(señales, etiquetas):
        N = len(s)
        
        # Cálculo de la FFT para señales reales y sus frecuencias correspondientes
        fft_vals = np.fft.rfft(s)
        frecuencias = np.fft.rfftfreq(N, 1/fs)
        
        # Magnitud normalizada para recuperar la amplitud real de los componentes sinusoides
        magnitud = np.abs(fft_vals) * (2.0 / N)
        
        # Corrección estricta para las componentes límites (Continua y Nyquist)
        magnitud[0] = magnitud[0] / 2.0
        if N % 2 == 0:
            magnitud[-1] = magnitud[-1] / 2.0

        ax.plot(frecuencias, magnitud, label=etiqueta)
        
    # Cambiamos a escala logarítmica
    ax.set_xscale('log')
    
    # Aplicamos límites
    if xlim is not None:
        ax.set_xlim(xlim)
        
    # Ticks para el grafico
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    
    ax.set_xticks(ticks_audio)
    ax.set_xticklabels(labels_audio)
    
    # Grilla doble (principal y secundaria) con líneas punteadas
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.legend(loc='upper right')
    
    plt.show()




"""
# Definir tasa de muestreo y número de muestras
fs = 44100
samples = 1000
t = np.arange(samples) / fs

# Señales sinusoidales de ejemplo
signal_1 = np.sin(2 * np.pi * 500 * t)
signal_2 = np.sin(2 * np.pi * 1000 * t)
"""
"""
signal_1, fs= sf.read("C:\\Users\\Familia\\Documents\\DONELLA 2026\\Graba iii\\BVox 1 RX_Melodyne.wav")
signal_2, fs= sf.read("C:\\Users\\Familia\\Documents\\DONELLA 2026\\Graba iii\\tp edicion\\PerdidosNaZonaSul_MeuBem_Full\\17_ElecGtr1.wav")


graficar_f(fs, [signal_1, signal_2], ["señal 1", "señal 2"], "espectros")
"""



