import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

n = np.linspace(0,63)
x = 3 * np.cos(2*np.pi*(n/10))
y = np.fft.fft(x)

# Graficar x e y
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n, x, marker='o')
plt.title('x(n)')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.abs(y), marker='o')
plt.title('Y(k)')
plt.xlabel('k')
plt.ylabel('Magnitud')
plt.grid(True)

plt.tight_layout()
plt.show()