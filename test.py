import matplotlib.pyplot as plt
import numpy as np
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, np.absolute(sp.real))
plt.show()

