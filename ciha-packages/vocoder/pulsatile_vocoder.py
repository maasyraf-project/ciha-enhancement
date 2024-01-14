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
        self.sampling_frequency = 48000
        self.weights = [0.98, 0.98, 0.98, 0.68, 0.68, 0.45, 0.45, 0.2, 0.2, 0.15, 0.15, 0.15]
        self.lp_frequency = 200
        self.pulse_length = 32e-6
        self.inter_pulse_gap = 2.1e-6
        self.electrode_selection_method = "sequential"
        self.pulse_per_second = 800
        self.B = 0.0156
        self.M = 1.5859
        self.compression_coefficient = 340.83
        self.TCL = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        self.MCL = [800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800]
        self.volume_gain = 1
        self.n_chan_interact = 5
        self.electrode_spacing = 0.0024
        self.decay_coef = 0.0036

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

        self.lenpulse = np.ceil((2 * self.pulse_length + self.ipg) * self.fs)
        self.block_delay = self.calculate_delay()

        # compression and conversion parameter
        self.B = params.B
        self.M = params.M
        self.alpha = params.compression_coefficient
        self.MCL = params.MCL
        self.TCL = params.TCL
        self.volume = params.volume_gain

        # auralization parameter
        self.m = params.n_chan_interact
        self.decay = params.decay_coef
        self.d = params.electrode_spacing

        # process the signal
        # (1) apply analysis filter
        self.output = self.analysis_filter(input_signal)

        # (2) obtain envelope and fine structure
        self.env, self.fine = self.feature_extraction(self.output)

        # (3) apply pulsatile sampling
        self.electrodogram = self.method_selection(self.env, self.fine)

        # (4) apply compression on obtained electrodogram
        self.electrodogram = self.compress(self.electrodogram)

        # (5) convert electrodogram to electrical domain
        self.electrodogram = self.convert_to_electrical(self.electrodogram)

        # (6) invers the electrical domain signal into acoutical, dan decompress the signal
        self.auralized_signal = self.convert_to_acoustic(self.electrodogram)
        self.auralized_signal = self.decompress(self.auralized_signal)

        # (7) calculate channel interaction and current spreading
        self.auralized_signal = self.channel_interaction(self.auralized_signal)

        # (8) synthesis using Gammatone filterbank
        self.vocoded_signal = self.synthesis_filter(self.auralized_signal)

    def synthesis_filter(self, input_signal: ndarray) -> np.ndarray:
        '''
        The function has objective to auralized signal and apply Gammatone filterbank to convert multi-channel signal into single-channel signal
        '''
        # create gammatone filterbank
        filterbank = gammatone_filterbank(
            self.order,
            self.fs,
            self.cf,
            False)

        # process the signal
        output = filterbank.process_filtering(input_signal)

        # set RMS of signal on each channel
        output = self.adjust_rms(output, method = 'linear')

        # create synthesizer filterbank



        return output

    def adjust_rms(self, input_signal: ndarray, method: str) -> np.ndarray:
        # define rms approach
        if method == 'linear':
            rms_factor = self.rms_per_channel
        elif method == 'dB':
            rms_factor = [pow(10, x/20) for x in self.rms_per_channel]

        # calculate current rms of input signal
        rms_input = np.real([np.sqrt(np.mean(np.dot(input_signal[x, :], np.conjugate(input_signal[x, :])))) for x in range(np.shape(input_signal)[0])])

        # calcualte linear gain, ratio between current and target RMS
        gain = rms_factor / rms_input

        # calcualte output
        output = [input_signal[x,:] * gain[x] for x in range(np.shape(input_signal)[0])]

        return output

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

        self.rms_per_channel = np.sqrt(np.mean(np.multiply(env, np.conjugate(env)), axis=1))

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
        if self.vocoder_type == "CIS":
            electrodogram = self.sampling_CIS(envelope)
            electrodogram = self.assert_sequential_stimulation(electrodogram)
        elif self.vocoder_type == "FSP":
            electrodogram = self.sampling_FSP(envelope, fine_structure)

        return electrodogram

    def sampling_CIS(self, envelope):
        # create output array
        sampled_pulse = np.zeros(np.shape(envelope))

        # definei intial parameter
        pulse = np.ones(1, self.lenpulse)
        n_samples = 0
        pps_idx = 0
        channel_shift_idx = 0

        # calculate channel shift
        if len(self.block_delay) > 1:
            channel_shift = [np.round((self.block_delay[idx] - self.lenpulse * self.nChan)/self.nChan) for idx in range(len(self.block_delay))]
        else:
            channel_shift = np.round((self.block_delay - self.lenpulse * self.nChan)/self.nChan)

        if self.nChan * self.lenpulse > np.min(self.block_delay):
            print('Using this pps, number of channels, pulselength, and fs would result in parallel stimulation! Please adjust these three factors, so that the followind equation is satisfied, with x = [1,2,3,4,...,100]: ((x * M) + pulselength_samples)*pps*M/M = fs');

        # conduct pulsatile sampling
        length_signal = np.shape(envelope)[1]

        while n_samples <= length_signal-1:
            channel_order = self.sort_electrode(np.linspace(0, np.shape(envelope)[0]-1, np.shape(envelope)[0])).astype(np.int64)

            for chan in range(self.nChan):
                # stop before last sample
                if n_samples > length_signal - self.lenpulse:
                    break

                # pulsatile sampling
                if envelope[channel_order[chan]][n_samples] > 0:
                    sampled_pulse[channel_order[chan]][n_samples : int(n_samples + self.lenpulse - 1)] = np.multiply(envelope[channel_order[chan]][n_samples], pulse)

                if envelope[channel_order[chan]][n_samples] > 0 and chan < np.shape(envelope)[0]:
                    # shift to the next channel
                    n_samples = int(n_samples + self.lenpulse + channel_shift[channel_shift_idx])

                    if np.size(channel_shift_idx) != 1:
                        channel_shift_idx += 1

                    channel_shift_idx = 0

                if n_samples == length_signal-1:
                    n_samples += 1
                    break

            n_samples = int(n_samples + self.block_delay[pps_idx] - (self.nChan-1) * (self.lenpulse + channel_shift[channel_shift_idx]))

            if pps_idx != len(self.block_delay)-1:
                pps_idx += 1

            pps_idx == 0


        return sampled_pulse

    def assert_sequential_stimulation(self, electrodogram):
        stimulus_pattern = np.zeros(np.shape(electrodogram))
        stimulus_pattern[electrodogram > 0] = 1

        number_of_stimulated_chan = np.sum(stimulus_pattern, 0)

        if any(number_of_stimulated_chan > 1):
            print("Warning, parallel electrode stimulation is detected! The higher amplitude will be preserved")

        index = np.column_stack(np.where(number_of_stimulated_chan > 1))

        for i in range(len(electrodogram[1][number_of_stimulated_chan > 1])):
            segment = electrodogram[:][index[i][1]]
            max_val = np.max(segment)
            max_idx = np.argmax(segment)
            segment[:] = 0
            segment[max_idx] = max_val
            electrodogram[:][index[i][1]] = segment

        return electrodogram

    def sampling_FSP(self, envelope, fine_structure):
        return None

    def sort_electrode(self, channel_array: ndarray):
        # select sorting method on electrde block stimulation
        if self.elec_method == "sequential":
            order = np.sort(channel_array)[::-1]

        return order

    def compress(self, input_electrodogram):
        output_electrodogram = np.zeros(np.shape(input_electrodogram))

        output_electrodogram[input_electrodogram < self.B] = 0

        index = np.column_stack(np.where((input_electrodogram >= self.B) & (input_electrodogram < self.M)))

        for idx in index:
            chan = idx[0]
            sample = idx[1]
            output_electrodogram[chan][sample] = np.log(1 + self.alpha * ((input_electrodogram[chan][sample] - self.B) / (self.M - self.B))) / np.log(1 + self.alpha)

        output_electrodogram[input_electrodogram >= self.M] = 1

        return output_electrodogram

    def decompress(self, input_electrodogram):
        output_electrodogram = np.zeros(np.shape(input_electrodogram))

        for chan in range(np.shape(input_electrodogram)[0]):
            for sample in range(np.shape(input_electrodogram)[1]):
                output_electrodogram[chan][sample] = (np.exp(np.log(1 + self.alpha) * (input_electrodogram[chan][sample])) - 1) / self.alpha * (self.M - self.B) + self.B

        output_electrodogram[input_electrodogram == 0] = 0
        return output_electrodogram

    def convert_to_electrical(self, input_electrodogram):
        output_electrodogram = np.zeros(np.shape(input_electrodogram))

        for chan in range(np.shape(input_electrodogram)[0]):
            for idx in range(np.shape(input_electrodogram)[1]):
                output_electrodogram[chan][idx] = (input_electrodogram[chan][idx] * self.volume * (self.MCL[chan] - self.TCL[chan])) + self.TCL[chan]

        for i in range(np.shape(input_electrodogram)[1]):
            idx = [input_electrodogram[:,i] == self.TCL]
            for j, bool in enumerate(idx):
                if bool[j] == True:
                    output_electrodogram[j, i] = 0

        return output_electrodogram

    def convert_to_acoustic(self, input_electrodogram):
        output_electrodogram = np.zeros(np.shape(input_electrodogram))

        for chan in range(np.shape(input_electrodogram)[0]):
            for idx in range(np.shape(input_electrodogram)[1]):
                output_electrodogram[chan][idx] = input_electrodogram[chan][idx] - self.TCL[chan]

        for i in range(np.shape(input_electrodogram)[1]):
            idx = [input_electrodogram[:,i] == np.multiply(-1, self.TCL)]
            for j, bool in enumerate(idx):
                if bool[j] == True:
                    # prevent overshoot
                    output_electrodogram[j, i] = 0

        for chan in range(np.shape(input_electrodogram)[0]):
            for idx in range(np.shape(input_electrodogram)[1]):
                output_electrodogram[chan][idx] = output_electrodogram[chan][idx]  / (self.volume * (self.MCL[chan] - self.TCL[chan]))

        return output_electrodogram

    def channel_interaction(self, input_electrodogram):
        weight = np.eye(self.nChan, self.nChan) * 0.5
        coef_right = [np.exp(-i * self.d/self.decay) for i in range(self.m)]
        coef_left = coef_right[::-1]

        for i in range(np.shape(weight)[0]):
            if self.m+i <= np.shape(weight)[0]:
                weight[i, i:self.m+i] = coef_right
            elif i == np.shape(weight)[0]-1:
                weight[i, i:i] = coef_right[0]
            else:
                len_coef = self.nChan - (i + 1)
                coef = coef_right[0:len_coef-1]
                weight[i, i:i+len_coef-1] = coef

        weight = weight[0:self.nChan, 0:self.nChan]

        weight = np.triu(weight, -1).T + weight

        # apply interaction
        idx_pulse = np.abs(input_electrodogram) > 0
        output_electrodogram = np.zeros(np.shape(input_electrodogram))

        for i in range(np.shape(input_electrodogram)[0]):
            idx_active_channel = idx_pulse[i, :] > 0
            interaction = np.outer(input_electrodogram[i, idx_active_channel] , weight[i, :].T)
            output_electrodogram[:, idx_active_channel] = output_electrodogram[:, idx_active_channel] + interaction.T

        return output_electrodogram

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
    #print(np.shape(output_signal.env))
    #output_signal.plot_channel(output_signal.env, "Plot signal per channel", "Sample", "Amplitude")
