import numpy as np
import soundfile as sf
import os

def cargar_audios_por_tipo(archivos_por_tipo):
    """
    Carga archivos .wav agrupados por tipo (como 'voces', 'instrumentos', etc.)

    Parámetro:
    - archivos_por_tipo (dict): un diccionario cuyas claves son tipos de audio 
      (por ejemplo, 'respuestas al impulso'), y los valores son listas de nombres de archivos .wav.

    Devuelve:
    - Un diccionario con los mismos tipos como claves, y una lista de arrays como valores.
    """
    
    resultado = {} 

    for tipo in archivos_por_tipo:
        lista_de_archivos = archivos_por_tipo[tipo]
        resultado[tipo] = []  # Creamos una lista vacía para este tipo

        for nombre_archivo in lista_de_archivos:
            # Verificamos si el archivo existe y termina en .wav
            if os.path.exists(nombre_archivo) and nombre_archivo.endswith(".wav"):
                
                audio, fs = sf.read(nombre_archivo)
                
                if audio.ndim > 1:  # Si el audio tiene más de un canal
                
                    audio = np.mean(audio, axis=1) # Convertir a mono 
                # Guardamos el audio y su frecuencia de muestreo como una tupla
                resultado[tipo].append((audio, fs))
            else:
                print(f"No se pudo cargar: {nombre_archivo} porque no existe o no es un archivo .wav")

    return resultado

"""
data = cargar_audios_por_tipo({
    'respuesta impulso': ['trabajo_practico/TP/ir_centre_stalls.wav', 'trabajo_practico/TP/1st_baptist_nashville_balcony.wav']
})
"""
