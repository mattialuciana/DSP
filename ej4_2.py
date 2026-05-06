import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

x, fs = sf.read("archivo_ejc3.wav")
print(fs)

t = np.arange(0, len(x)/fs, 1/fs)
plt.figure(figsize=(10, 6))
plt.plot(t, x, '.')
plt.title("Señal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show() 

inicio = np.nonzero(x)[0][0]
final = int(np.round(0.005 * fs))

print("inicio ", inicio)
print("final ", final)

x_nueva=x[inicio:final+1]
t_nueva = np.arange(0, len(x_nueva)/fs, 1/fs)
plt.figure(figsize=(10, 6))
plt.plot(t_nueva, x_nueva, '.')
plt.title("Señal nueva")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show() 