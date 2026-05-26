
# GRAFICAR FUNCIONES EN EL TIEMPO. revisar los rastros de copilot

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def graficar_t(fs, señales, etiquetas=None, titulo='Señales en el Tiempo'):
    """
    Grafica una o varias señales en función del tiempo calculando 
    la duración automáticamente.
    
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
        # Calculamos la cantidad de muestras de ESTA señal
        muestras = len(s) 
        
        # Calculamos la duración exacta de esta señal
        duracion = muestras / fs 
        
        # Generamos el vector de tiempo correspondiente
        t = np.arange(0, duracion, 1/fs)
        
        # Graficamos asegurando que coincidan las longitudes (por si acaso)
        ax.plot(t[:muestras], s, label=etiqueta)
    

    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.legend(loc='upper right') 
    ax.grid(True)
    
    plt.show()
    return fig
    


# GRAFICAR FUNCIONES EN FRECUENCIA. revisar los rastros de copilot. cuidado con la escala logarítmica  (límites?)
def graficar_f(fs, señales, etiquetas=None, titulo='Espectro de Amplitud (Fourier)'):
    """
    Calcula y grafica el espectro de magnitud de Fourier de una o varias señales.
    
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
    """
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    if etiquetas is None:
        etiquetas = [f'Espectro {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determinar el menor valor de frecuencia > 0 para linthresh de symlog
    min_nonzero_freq = np.inf

    for s, etiqueta in zip(señales, etiquetas):
        N = len(s)
        
        # Cálculo de la FFT para señales reales y sus frecuencias correspondientes
        fft_vals = np.fft.rfft(s)
        frecuencias = np.fft.rfftfreq(N, 1/fs)
        
        # Magnitud normalizada para recuperar la amplitud real de los componentes sinusoides
        magnitud = np.abs(fft_vals) * (2.0 / N)
        
        # Corrección estricta para las componentes límites (Continua y Nyquist) que no se duplican
        magnitud[0] = magnitud[0] / 2.0
        if N % 2 == 0:
            magnitud[-1] = magnitud[-1] / 2.0
            
        # Registrar la menor frecuencia positiva encontrada
        mask = frecuencias > 0
        if np.any(mask):
            min_nonzero_freq = min(min_nonzero_freq, frecuencias[mask].min())

        # Trazar todo (incluye f=0). Usaremos una escala symlog más abajo.
        ax.plot(frecuencias, magnitud, label=etiqueta)
        
    ax.set_title(titulo)
    # Usar escala symlog para mantener f=0 y comportamiento log fuera de linthresh
    if np.isfinite(min_nonzero_freq) and min_nonzero_freq > 0:
        ax.set_xscale('symlog', linthresh=min_nonzero_freq)
        ax.set_xlabel(f"Frecuencia (Hz)")
    else:
        ax.set_xscale('linear')
        ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.legend(loc='upper right')
    ax.grid(True)
    
    plt.show()
    return fig





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



