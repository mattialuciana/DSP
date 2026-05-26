import numpy as np

def generar_filtros_media_movil(M):
    """
    Genera las respuestas al impulso para un filtro de media móvil
    de 1, 2 y 3 pasadas.
    
    Parámetros:
    M (int): El largo de la ventana del filtro.
    
    Retorna:
    tuple: (h1, h2, h3) que son los arrays (numpy arrays) de las respuestas al impulso.
    """
    
    # 1. Primera pasada: h[n] = 1/M
    # Creamos un arreglo de tamaño M donde cada valor es 1/M
    h1 = np.ones(M) / M
    
    # 2. Segunda pasada: Convolución de h1 consigo misma
    h2 = np.convolve(h1, h1)
    
    # 3. Tercera pasada: Convolución de h2 con h1
    h3 = np.convolve(h2, h1)
    
    return h1, h2, h3