# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import math
from scipy.signal import lfilter, lfilter_zi, butter
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
        self.lp_frequency = 200
        self.pulse_length = 32e-6
        self.inter_pulse_gap = 2.1e-6
        self.electrode_selection_method = "sequential"
        self.pulse_per_second = 800

class pulsatile_vocoder():
    '''
    The implementation of pulsatilve vocoder, as described in Brecker (2009)
    '''
    def __init__(self, input_signal: np.ndarray, params: tuple, method: str) -> np.ndarray:
        super(pulsatile_vocoder, self).__init__()

        # filterbank parameter
        self.order = params.filterbank_order
        self.fs = params.sampling_frequency
        self.weights = params.weights

        # vocoder parameter
        self.nChan = params.number_of_channels
        self.cf = params.center_frequencies
        self.lp_filter_frequency = params.lp_frequency
        self.vocoder_type = method

        # sampling parameter
        self.pulse_length = params.pulse_length
        self.ipg = params.inter_pulse_gap
        self.pps = 800
        self.elec_method = params.electrode_selection_method

        self.lenpulse = np.ceil(2 * self.pulse_length * self.ipg * self.fs)
        self.block_delay = self.calculate_delay()

        print(self.block_delay)

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

    def feature_extraction(self, input_signal:ndarray) -> tuple:
        '''
        This function used for obtaining envelope and fine structure of filtered signals
        '''
        # obtain envelope including weights on each channel
        env = [np.sqrt(np.multiply(self.weights[chan], pow(np.real(input_signal[chan][:]),2) +
            pow(np.imag(input_signal[chan][:]), 2)))
            for chan in range(np.shape(input_signal)[0])]

        env = self.lp_filter(env)

        # obtain fine structure
        fine = np.real(input_signal)

        return env, fine

    def lp_filter(self, envelope: ndarray) -> ndarray:
        '''
        This function used for applying low pass filter on signal's envelope
        '''
        # create filter coefficient
        b, a = butter(1, self.lp_filter_frequency / (self.fs/2))

        # applying the filter
        filtered_envelope = [lfilter(b, a, envelope[chan]) for chan in range(np.shape(envelope)[0])]

        return filtered_envelope

    def calculate_delay(self):
        # calculate block delay during sampling
        delay = self.fs / self.pps
        if delay < 1:
            print("The pps is too high")

        delay_block = [np.ceil(self.fs/self.pps)]
        while np.round(np.mean(delay_block) * 1e5) != np.round(delay*1e5):
            diff = delay - delay_block
            if diff > 0:
                delay_block = delay_block.append(np.ceil(fs/pps))
            else:
                delay_block = delay_block.append(np.floor(fs/pps))

        if len(delay_block) > 1:
            print("pps * M is not a divisor of fs. Make sure that the following equation with x = [1,2,3,4,...,100 is satisfied: x*pps*M = fs")

        return delay_block

    def method_selection(self, envelope, fine_structure):
        if method == "CIS":
            electrodogram = sampling_CIS(envelope)
        elif method == "FSP":
            electrodogram = sampling_FSP(envelope, fine_structure)

        return electrodogram

    def sampling_CIS(self, envelope):
        # create output array
        sampled_pulse = np.zeros(np.shape(envelope))

        # definei intial parameter
        pulse = np.ones(1, self.lenpulse)
        n_samples = 1
        pps_idx = 1
        channel_shift_idx = 1

        # calculate channel shift
        if len(self.block_delay) > 1:
            channel_shift = [np.sound((self.block_delay[idx] - self.lenpulse * self.nChan)/self.nChan) for idx in range(len(self.block_delay))]
        else:
            channel_shift = np.sound((self.block_delay[idx] - self.lenpulse * self.nChan)/self.nChan)

        if self.nChan * self.lenpulse > np.min(self.block_delay):
            print('Using this pps, number of channels, pulselength, and fs would result in parallel stimulation! Please adjust these three factors, so that the followind equation is satisfied, with x = [1,2,3,4,...,100]: ((x * M) + pulselength_samples)*pps*M/M = fs');



        return None

    def plot_channel(self, input_signal: ndarray, title: str, xlabel: str, ylabel:str) -> None:
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
    output_signal = pulsatile_vocoder(signal, pulsatile_params, "CIS")
    print(np.shape(output_signal.env))
    output_signal.plot_channel(output_signal.env, "Plot signal per channel", "Sample", "Amplitude")
