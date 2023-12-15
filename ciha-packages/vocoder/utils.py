# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import math
from scipy.signal import lfilter, lfilter_zi
from scipy.fft import fft
import matplotlib.pyplot as plt

class gammatone_filterbank():
    '''
    Implementation of computationally efficient of gammatone filterbank with equivalent rectangular bandwidth (ERB). Please refers to Hohmann (2002) for details of this implementation
    '''

    # global parameter
    L = 24.7
    Q = 9.265

    # initiation function
    def __init__(self, filter_order: int, frequency_sampling: int, center_frequencies: np.ndarray, verbose: bool) -> None:
        super(gammatone_filterbank, self).__init__()

        # filter parameter
        self.order = filter_order
        self.fs = frequency_sampling
        self.cf = center_frequencies
        self.filterbank_coefficient = np.zeros((len(self.cf),1), dtype="complex_")
        self.filterbank_norm_factor = np.zeros((len(self.cf),1), dtype="float")
        self.filterbank_state = np.zeros((len(self.cf), self.order), dtype="float")

        # obtain filter coeffiecient and normalization factor
        self.filterbank_coefficient, self.filterbank_norm_factor , self.filterbank_state = self.create_filter()

        # show impulse response of filterbank
        if verbose:
            self.ir_plot()

    def create_filter(self: tuple) -> tuple:
        # convert center frequencies from Hz to ERB scale
        # equation no. 16 in Hohmann (2002)
        # freq_erb = [ self.Q * np.log10(1 + x / (self.L * self.Q)) for x in self.cf ]

        # calculate filter coefficient and its normalization factor for each center frequencies
        for i in range(len(self.cf)):
            coef, nfactor, state = self.calculate_filter_params(self.cf[i])
            self.filterbank_coefficient[i] = coef
            self.filterbank_norm_factor[i] = nfactor
            self.filterbank_state[i][:] = state

        return self.filterbank_coefficient, self.filterbank_norm_factor, self.filterbank_state

    def calculate_filter_params(self, freq: int) -> np.ndarray:
        # equation no. 13 in Hohmann (2002)
        erb_aud_filters = (self.L + freq / self.Q)

        # equation no. 14 in Hohmann (2002)
        a_gamma = (np.pi * math.factorial(2*self.order - 2)
            * pow(2, -(2*self.order - 2))
            / pow(math.factorial(self.order - 1),2))
        b = erb_aud_filters / a_gamma
        damping = np.exp(-2 * np.pi * b / self.fs)

        # equation no. 10 in Hohmann (2002)
        beta = 2 * np.pi * freq / self.fs

        # equation no. 1 in Hohmann (2002)
        analog_coefficient = damping * np.exp(1j * beta)
        # coefficients = pow(number_of_sample, gamma - 1) * analog_coefficient

        # section 2.4 in Hohmann (2002)
        norm_factor = 2 * pow((1 - np.abs(analog_coefficient)), 4)

        # create filter state
        state = np.zeros((1, self.order))

        return analog_coefficient, norm_factor, state

    def ir_plot(self: tuple) -> None:
        len_signal = 8192
        output = np.zeros((len(self.cf), len_signal), dtype="complex_")

        for i in range(len(self.cf)):
            impulse = np.zeros((len_signal-1, 1))
            impulse = np.insert(impulse, 0, 1)

            b = self.filterbank_norm_factor[i]
            a = self.filterbank_coefficient[i]
            a = np.insert(-a, 0, 1)

            for j in range(self.order):
                filtered_impulse = lfilter(b, a, impulse)
                b = [1]
                impulse = filtered_impulse

            output[i][:] = filtered_impulse


        # plot the figure
        freq = np.arange(len(impulse)) * self.fs / len(impulse)
        plt.figure()
        for i in range(len(self.cf)):
            freq_response = fft(np.real(output[i]))
            freq_response = 20 * np.log10(abs(freq_response))
            plt.plot(freq, freq_response)
        plt.ylim([-50, 0])
        plt.xlim([0, 8000])
        plt.show()

    def process_filtering(self: tuple, input_signal: np.ndarray) -> np.ndarray:
        len_signal = len(input_signal)
        output = np.zeros((len(self.cf), len_signal), dtype="complex_")

        for i in range(len(self.cf)):
            impulse = input_signal

            b = self.filterbank_norm_factor[i]
            a = self.filterbank_coefficient[i]
            a = np.insert(-a, 0, 1)

            for j in range(self.order):
                filtered_impulse = lfilter(b, a, impulse)
                b = [1]
                impulse = filtered_impulse

            output[i][:] = filtered_impulse

        return output

# if __name__ == "__main__":
    # cf = [120, 235, 384, 579, 836, 1175, 1624, 2222, 3019, 4084, 5507, 7410]
    # gamma_filt = gammatone_filterbank(4,
    #        16000,
    #        [120, 235, 384, 579, 836, 1175, 1624, 2222, 3019, 4084, 5507, 7410],
    #        True
    #        )