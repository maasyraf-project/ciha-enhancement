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
    gain_calc = 1
    iteration = 4

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

    def create_synthesizer(self:tuple, delay:int) -> None:
        delay_samples = int(np.round(delay * self.fs))

        self.create_delay(delay_samples)
        self.create_mixer()

    def create_delay(self:tuple, delay_samples:int) -> None:
        # cerate impulse
        impulse = np.zeros((delay_samples+2,))
        impulse[0] = 1

        # create impulse response
        impulse_response = self.process_filtering(impulse)
        impulse_response = np.abs(impulse_response)

        # identify maximum values from impulse response array
        max_ir_idx = np.argmax(impulse_response, axis=0)
        max_ir_idx = max_ir_idx - (max_ir_idx > delay_samples + 1)

        self.delay = delay_samples + 1 - max_ir_idx

        self.delay_memory = np.zeros((len(self.cf), np.max(self.delay)), dtype="complex")

        # calculate slopes
        slopes = np.zeros((len(self.cf),))
        for chan in range(len(self.cf)):
            max_chan_idx = max_ir_idx[chan]
            slopes[chan] = (impulse_response[chan, max_chan_idx+1] - impulse_response[chan, max_chan_idx-1])

        slopes = slopes / np.abs(slopes)

        self.delay_phase_factor = [1j / slopes[i] for i in range(len(slopes))]

    def create_mixer(self:tuple) -> None:
        # create ERB filter
        ERB = self.L + self.cf[0] / self.Q
        a_gamma = (np.pi * math.factorial(2*self.order - 2)
            * pow(2, -(2*self.order - 2)))
        c_gamma = 2 * np.sqrt(pow(2, 1/self.order)-1)
        bandwidth = c_gamma / a_gamma * ERB
        low_cutoff_freq = self.cf[0] - np.floor(1*bandwidth/2)

        samples = pow(2, np.ceil(np.log2(self.gain_calc * self.fs / low_cutoff_freq))).astype("int")

        # create impulse
        impulse = np.zeros((samples,))
        impulse[0] = 1

        # spectrum analysis
        spec_idx = [int(np.round(self.cf[x]) * samples / self.fs) for x in range(len(self.cf))]
        self.mixer_gain = np.ones((len(self.cf,)))

        impulse_response = self.process_filtering(impulse)
        impulse_response = self.process_delay(impulse_response)

        # obtain spectrum response
        ir_spectrum = [np.fft.fft(np.real(impulse_response[x])) for x in range(np.shape(impulse_response)[0])]

        # calculate mixer gain
        for i in range(self.iteration):
            true_spectrum = [ir_spectrum[x][spec_idx[x]] for x in range(len(self.cf))]
            true_spectrum = np.dot(true_spectrum, self.mixer_gain)

            self.mixer_gain = self.mixer_gain / (np.abs(true_spectrum))

    def process_delay(self:tuple, input:np.ndarray) -> np.ndarray:
        output = np.zeros((len(self.cf), np.shape(input)[1]), dtype="complex")

        for chan in range(len(self.cf)):
            if self.delay[chan] == 0:
                output[chan, :] = np.real(input[chan, :]) * self.delay_phase_factor[chan]
            else:
                tmp_out = np.concatenate((self.delay_memory[chan][0:self.delay[chan]-1], np.real(input[chan][:]) * self.delay_phase_factor[chan]))
                self.delay_memory[chan, 0:self.delay[chan]-1] = tmp_out[np.shape(input)[1]:len(tmp_out)]
                output[chan,:] = tmp_out[0:np.shape(input)[1]]

        return output

    def process_mixer(self:tuple, input:np.ndarray) -> np.ndarray:
        # apply mixer gain into input
        output = input * self.mixer_gain[:, None]

        return output

    def process_synthesis(self:tuple, input: np.ndarray) -> np.ndarray:
        # delay process
        output = self.process_delay(input)

        # mixing process
        output = self.process_mixer(output)

        return output

    def process_filtering(self: tuple, input_signal: np.ndarray) -> np.ndarray:

        if input_signal.ndim == 1:
            len_signal = len(input_signal)
            input_signal = np.expand_dims(input_signal, axis=0)

        if np.shape(input_signal)[0] == 1:
            idx_input = np.zeros((len(self.cf),), dtype=int)
        elif np.shape(input_signal)[0] == len(self.cf):
            idx_input = np.linspace(0, 11, num=12).astype(int)
            len_signal = np.shape(input_signal)[1]

        output = np.zeros((len(self.cf), len_signal), dtype="complex_")

        for i in range(len(self.cf)):
            impulse = input_signal[idx_input[i], :]

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