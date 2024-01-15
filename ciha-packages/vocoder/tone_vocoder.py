# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from numpy.matlib import repmat
import math
from scipy.signal import lfilter, lfilter_zi, butter
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt

class vocoder_params():
    '''
    A class for initiate parameter of vocoder
    '''
    def __init__(self) -> tuple:
        super(vocoder_params, self).__init__()

        # initiation of vocoder parameter
        self.number_of_channels = 2
        self.center_frequencies = [120, 235, 384, 579, 836, 1175, 1624, 2222, 3019, 4084, 5507, 7410]
        self.sampling_frequency = 48000
        self.pre_emphasis_coef = 0.9378


class tone_vocoder():
    '''
    The implementation of pulsatilve vocoder, as described in Brecker (2009)
    '''
    def __init__(self, input_signal: np.ndarray, params: tuple, method: str) -> np.ndarray:
        super(tone_vocoder, self).__init__()

        # vocoder parameter
        self.nchan = params.number_of_channels
        self.fs = params.sampling_frequency
        self.cf = params.center_frequencies
        self.pre_emphasis_coef = params.pre_emphasis_coef

        # (1) pre-emphasis stage
        self.output = self.pre_emphasis_filter(input_signal)

        # (2) apply filterbank on signal
        self.output = self.decompose(self.output)

        print("process done")


    def pre_emphasis_filter(self:tuple, input:np.ndarray) -> np.ndarray:
        # apply pre-emphasis filter
        output = lfilter([1, -self.pre_emphasis_coef], 1, input)

        return output

    def decompose(self:tuple, input:np.ndarray) -> np.ndarray:
        # create filterbank
        filterbank = self.create_filterbank(self.nchan)

    def create_filterbank(self:tuple, n_channels):
        # calculate filter coefficient
        if self.nchan == 1:
            Wn = repmat([400], 1, 2)
            Bw = np.multiply(0.5, [-7500, 7500]) / (self.fs/2)
        elif self.nchan == 2:
            Wn = repmat([[792], [3392]], 1, 2)
            Bw = np.multiply(0.5, [[-984, 984], [-4215, 4215]]) / (self.fs/2)

        Wn = Wn / (self.fs/2)
        Bw = Bw / (self.fs/2)

        Wn = Wn + Bw
        Wn[Wn>1] = 0.99
        Wn[Wn<0] = 0.01

        return filterbank


if __name__ == "__main__":
    # define input signal
    rate, signal = wavfile.read("example.wav")

    # apply pulsatile vocoder
    pulsatile_params = vocoder_params()
    pulsatile_params.center_frequencies = rate
    output_signal = tone_vocoder(signal, pulsatile_params, "CIS")
