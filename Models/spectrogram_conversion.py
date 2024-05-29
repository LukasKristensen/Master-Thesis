from scipy.signal import spectrogram
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def convert_to_spectrogram(file_path, save_path):
    fs, data = wavfile.read(file_path)
    f, t, Sxx = spectrogram(data, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()


models = ['AE25', 'AE100', 'AEVQ']


for model in models:
    for file in os.listdir(f'../Results/{model}'):
        if file.endswith('.wav') and 'spectrogram' not in file:
            convert_to_spectrogram(f'../Results/{model}/{file}', f'../Results/{model}/{file[:-4]}_spectrogram.png')
