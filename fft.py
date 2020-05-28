from scipy.signal import butter, lfilter
import scipy.fftpack as fftpack
import numpy as np
import padasip as pa
from scipy.signal import resample

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def window_filter(data, lc, uc, fs, return_time_domain=True):
    freq, rfft = real_fft(data, fs)
    # freq, rfft = resample(freq, 2000), resample(rfft, 2000)
    rfft[(freq >= uc) | (freq <= lc)] = 0
    if return_time_domain:
        return np.fft.irfft(rfft)
    else:
        return freq, np.abs(rfft)


def real_fft(data, fs):
    freq = np.fft.rfftfreq(len(data), d=1/fs)
    return freq, np.fft.rfft(data)


def get_pca_from_signals(signals):
    return np.abs(pa.preprocess.pca.PCA(np.asarray(signals).T, n=1))
    # pca_sgnl = np.transpose([buffer.buffer_key_array for buffer in self.buffer.values()])
    # self.pca = pa.preprocess.pca.PCA(pca_sgnl, n=1)
    # pass






