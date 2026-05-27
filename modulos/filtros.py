import numpy as np
from graficos import graficar_t, graficar_f, graficar_analisis
from respuestaf import respuesta_f
from senales_temporales import generar_tono_puro


def filtros_media_movil(M,N):
    """
    Genera las respuestas al impulso para un filtro de media móvil
    de 1, 2 y 3 pasadas.
    
    Parámetros:
    M (int): El largo de la ventana del filtro.
    N (int): La longitud de la respuesta al impulso.
    
    Retorna:
    tuple: (h1, h2, h3) que son los arrays (numpy arrays) de las respuestas al impulso.
    """
    
    delta = np.zeros(N)
    delta[0] = 1

    # 1. Primera pasada: h[n] = 1/M
    # Creamos un arreglo de tamaño M donde cada valor es 1/M
    p1 = np.ones(M) / M
    
    # 2. Segunda pasada: Convolución de h1 consigo misma
    p2 = np.convolve(p1, p1)
    
    # 3. Tercera pasada: Convolución de h2 con h1
    p3 = np.convolve(p2, p1)

    h1 = np.convolve(delta, p1)[:N]
    h2 = np.convolve(delta, p2)[:N]
    h3 = np.convolve(delta, p3)[:N]
    
    return h1, h2, h3

def filtro_peine(b0, b1, b2, N):
    """
    Genera la respuesta al impulso h[n] para un filtro peine FIR.
    
    Parámetros
    ----------
    b0, b1, b2 : float
        Coeficientes constantes del filtro.
        
    Retorna
    -------
    h : NumPy array
        Vector con la respuesta al impulso h[n].
    """

    
    delta = np.zeros(N)
    delta[0] = 1

    filtro = np.array([b0,b1,b2])
    h = np.convolve(delta, filtro)[:N]

    return h 

def filtro_fir(N, path="C:\\Users\\Administrator\\Documents\\untref\\DSP\\DSP\\archivos_tp1c2026\\fir_hamming_1000Hz.npy"):  
    #esto funciona solamente para mi compu pero bueno
    delta = np.zeros(N)
    delta[0] = 1
    
    coeficientes_fir = np.load(path)
    h = np.convolve(delta, coeficientes_fir)[:N]

    return h

def analisis_filtros(filtro, fs):
    """
    Toma la respuesta al impulso de un filtro, calcula su FFT real 
    y devuelve el vector de frecuencias, el módulo y la fase en radianes.
    
    Parámetros
    ----------
    filtro : NumPy array
        Respuesta al impulso del filtro (coeficientes).
    fs : int
        Frecuencia de muestreo en Hz.
        
    Retorna
    -------
    freqs : NumPy array (Vector de frecuencias en Hz)
    modulo : NumPy array (Magnitud o ganancia del filtro)
    fase_rad : NumPy array (Fase del filtro en radianes)
    """
    N = len(filtro)
    H_w = np.fft.rfft(filtro)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    modulo = np.abs(H_w)
    fase_rad = np.angle(H_w)

    return freqs, modulo, fase_rad


hp = filtro_peine(1,0.5,1,N=512)  
fcia, modulo, fase = analisis_filtros(hp,3000)
x=graficar_analisis(fcia, modulo, fase)

""""
hp = filtro_peine(1,0,0,N=512)  
fcia, modulo, fase = analisis_filtros(hp,1)
x=graficar_analisis(fcia, modulo, fase)
hp = filtro_peine(1,2,1,N=512)  
fcia, modulo, fase = analisis_filtros(hp,1)
x=graficar_analisis(fcia, modulo, fase)


h1, h2, h3 = filtros_media_movil(M=16, N=512)
fcia1, modulo1, fase1 = analisis_filtros(h1, 1)
fcia2, modulo2, fase2 = analisis_filtros(h2, 1)
fcia3, modulo3, fase3 = analisis_filtros(h3, 1)
x1=graficar_analisis(fcia1, modulo1, fase1)
x2=graficar_analisis(fcia2, modulo2, fase2)
x3=graficar_analisis(fcia3, modulo3, fase3)
"""


def filtrar_temporal(filtro, señal):
    """
    agregar docstring
    """
    señal_filtrada = np.convolve(señal, filtro, mode='same')

    return señal_filtrada

"""
tono_puro = generar_tono_puro(amplitud=0.5, duracion=1, fs=3000, frecuencia=1000)

señal_filtrada = filtrar_temporal (hp, tono_puro) [:len(tono_puro)]  
"""

def filtrar_frecuencial(filtro, señal, fs):
    """
    agregar docstring
    """
    H_w = np.fft.rfft(filtro, n=len(señal))
    X_w = np.fft.rfft(señal)
    Y_w = H_w * X_w

    señal_filtrada =np.fft.irfft(Y_w, n=len(señal))

    return señal_filtrada

"""
señal_filtrada = filtrar_frecuencial (hp, tono_puro, fs=3000) 
"""

señal_filtrada_1 = filtrar_temporal(hp, tono_puro)

graficar_f(fs=3000, señales=señal_filtrada_1, titulo="tono puro filtrado")

