# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import math
from scipy.signal import lfilter, lfilter_zi
from scipy.fft import fft
import matplotlib.pyplot as plt
from utils import gammatone_filterbank

class vocoder_params():
    '''
    A class for initiate parameter of vocoder
    '''
    def __init__(self) -> tuple:
        super(vocoder_params, self).__init__()

        # initiation of vocoder parameter
        self.number_of_channels = 12
        self.center_frequencies = [120, 235, 384, 579, 836, 1175, 1624, 2222, 3019, 4084, 5507, 7410]
        self.filterbank_order = 4
        self.sampling_frequency = 16000
        self.weights = [0.98, 0.98, 0.98, 0.68, 0.68, 0.45, 0.45, 0.2, 0.2, 0.15, 0.15, 0.15]


class pulsatile_vocoder():
    '''
    The implementation of pulsatilve vocoder, as described in Brecker (2009)
    '''
    def __init__(self, input_signal: np.ndarray, params: tuple) -> np.ndarray:
        super(pulsatile_vocoder, self).__init__()

        # filterbank parameter
        self.order = params.filterbank_order
        self.fs = params.sampling_frequency
        self.weights = params.weights

        # vocoder parameter
        self.nChan = params.number_of_channels
        self.cf = params.center_frequencies

        # process the signal
        # (1) apply analysis filter
        self.output = self.analysis_filter(input_signal)

        # (2) obtain envelope and fine structure
        self.env, self.fine = self.feature_extraction(self.output)


    def analysis_filter(self, input_signal: ndarray) -> np.ndarray:
        '''
        This function used for filter signal into 12-channels of subband signals using Gammatone filterbank
        '''
        # create gammatone filterbank
        filterbank = gammatone_filterbank(
            self.order,
            self.fs,
            self.cf,
            False)

        # process the signal with defined filterbank
        output = filterbank.process_filtering(input_signal)

        return output

    def feature_extraction(self, input_signal:ndarray) -> ndarray:
        '''
        This function used for obtaining envelope and fine structure of filtered signals
        '''
        # obtain envelope including weights on each channel
        env = [np.sqrt(np.multiply(self.weights[chan], pow(np.real(input_signal[chan][:]),2) + pow(np.imag(input_signal[chan][:]), 2))) for chan in range(np.shape(input_signal)[0])]

        # obtain fine structure
        fine = np.real(input_signal)

        return env, fine

    def plot_channel(self, input_signal: ndarray, title: str, xlabel: str, ylabel:str, ylim: ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8,8), nrows=12, ncols=1, sharey=True)
        plt.subplots_adjust(top=0.85, wspace=0.15)

        # creating a dictionary
        font = {'size': 7}
        plt.rc('font', **font)

        fig.text(0.5, 0.04,  xlabel, ha='center')
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
        fig.suptitle(title)

        for i in range(np.shape(input_signal)[0]):
            # ax[i] = plt.axes(frameon=False)
            ax[i].get_xaxis().tick_bottom()
            ax[i].plot(input_signal[i][:])

        plt.show()



if __name__ == "__main__":
    # define input signal
    signal = np.random.normal(0, 0.5, 80000)

    # apply pulsatile vocoder
    pulsatile_params = vocoder_params()
    output_signal = pulsatile_vocoder(signal, pulsatile_params)
    print(np.shape(output_signal.output))
    output_signal.plot_channel(output_signal.fine, "Plot signal per channel", "Sample", "Amplitude", [-0.5 , 0.5])
