# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from numpy.matlib import repmat
import math
from scipy.signal import lfilter, lfilter_zi, butter, filtfilt, freqz
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

class vocoder_params():
    '''
    A class for initiate parameter of vocoder
    '''
    def __init__(self) -> tuple:
        super(vocoder_params, self).__init__()

        # initiation of vocoder parameter
        self.number_of_channels = 5
        self.sampling_frequency = 48000
        self.pre_emphasis_coef = 0.9378
        self.cutoff_envelope_filter = 160
        self.verbose = True

class tone_vocoder():
    '''
    The implementation of pulsatilve vocoder, as described in Brecker (2009)
    '''
    def __init__(self, input_signal: np.ndarray, params: tuple, method: str) -> np.ndarray:
        super(tone_vocoder, self).__init__()

        # vocoder parameter
        self.nchan = params.number_of_channels
        self.fs = params.sampling_frequency
        self.pre_emphasis_coef = params.pre_emphasis_coef
        self.cutoff_env = params.cutoff_envelope_filter
        self.verbose = params.verbose

        # (1) pre-emphasis stage
        self.output = self.pre_emphasis_filter(input_signal)

        # (2) apply filterbank on signal
        self.env, self.output = self.vocode(self.output)

        # (3) adjusg the level same as input
        self.output = self.output * (np.sqrt(np.mean(input_signal**2)) / np.sqrt(np.mean(self.output**2)))

    def pre_emphasis_filter(self:tuple, input:np.ndarray) -> np.ndarray:
        # apply pre-emphasis filter
        b = [1, -self.pre_emphasis_coef]
        a = 1
        output = lfilter(b, a, input)

        # create figure of impulse response filter
        if self.verbose == True:
            amp, phase, freq = self.extract_ir(b, a)

            fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1)

            ax.semilogx(freq, amp, 'b')
            ax.set_title("Frequency response of pre-emphasis filter", pad=0)
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("magnitude (dB)", color='b')
            ax.grid(which="both", axis="both")

            ax2 = ax.twinx()
            ax2.plot(freq, phase, 'g')
            ax2.set_ylabel("phase (radian)", color='g')

            plt.show()


        return output

    def vocode(self:tuple, input:np.ndarray) -> np.ndarray:
        # create filterbank
        coef = self.create_filterbank(self.nchan)

        # create low pass filter to extract envelope on each channel
        b_env, a_env = butter(2, self.cutoff_env/(self.fs/2), "low")

        # create figure of impulse response filter
        if self.verbose == True:
            amp, phase, freq = self.extract_ir(b_env, a_env)

            fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1)

            ax.semilogx(freq, amp, 'b')
            ax.set_title("Frequency response of pre-emphasis filter", pad=0)
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("magnitude (dB)", color='b')
            ax.grid(which="both", axis="both")

            ax2 = ax.twinx()
            ax2.plot(freq, phase, 'g')
            ax2.set_ylabel("phase (radian)", color='g')

            plt.show()


        # create env and output array
        env = np.zeros((self.nchan, np.shape(input)[0]))
        vocoded = np.zeros(np.shape(input))

        if self.verbose == True:
            fig1, ax1 = plt.subplots(figsize=(6,4), nrows=1, ncols=1)

        for i in range(self.nchan):
            # create 4th-order Butterworth-filter
            b, a = butter(4, coef[i,:], "bandpass")

            # plot the impulse response of the filterbank
            # create figure of impulse response filter
            if self.verbose == True:
                amp, phase, freq = self.extract_ir(b, a)

                ax1.semilogx(freq, amp)
                ax1.set_title("Frequency response of pre-emphasis filter", pad=0)
                ax1.set_xlabel("frequency (Hz)")
                ax1.set_ylabel("magnitude (dB)", color='b')
                ax1.grid(which="both", axis="both")

                #ax2 = ax.twinx()
                #ax2.plot(freq, phase, 'g')
                #ax2.set_ylabel("phase (radian)", color='g')

            # apply filter on each channel
            output = filtfilt(b, a, input)

            # apply half-wave rectification
            output[output<0] = 0

            # apply low pass filter to obtain envelope
            env[i,:] = lfilter(b_env, a_env, output)

            # create tone mdulator
            f = np.exp(np.mean(np.log(coef[i,:]))) * (self.fs/2)
            tone = np.sin(2*np.pi*(f/self.fs)*np.arange(np.shape(env)[1]))
            voc = env[i,:] * tone

            # add to output
            vocoded = vocoded + voc

        plt.show()

        return env, vocoded

    def create_filterbank(self:tuple, n_channels):
        # calculate filter coefficient
        if self.nchan == 1:
            Wn = repmat([400], 1, 2)
            Bw = np.multiply(0.5, [-7500, 7500])
        elif self.nchan == 2:
            Wn = repmat([[792], [3392]], 1, 2)
            Bw = np.multiply(0.5, [[-984, 984], [-4215, 4215]])
        elif self.nchan == 3:
            Wn = repmat([[545], [1438], [3793]], 1, 2)
            Bw = np.multiply(0.5, [[-491, 491], [-1295, 1295], [-3414, 3414]])
        elif self.nchan == 4:
            Wn = repmat([[460], [953], [1971], [4078]], 1, 2)
            Bw = np.multiply(0.5, [[-321, 321], [-664,664], [-1373, 1373], [-2842, 2842]])
        elif self.nchan == 5:
            Wn = repmat([[418], [748], [1339], [2396], [4287]], 1, 2)
            Bw = np.multiply(0.5, [[-237, 237], [-423, 423], [-758, 758], [-1356, 1356], [-2426, 2426]])

        Wn = Wn / (self.fs/2)
        Bw = Bw / (self.fs/2)

        Wn = Wn + Bw
        Wn[Wn>1] = 0.99
        Wn[Wn<0] = 0.01

        return Wn

    def extract_ir(self:tuple, b:np.ndarray, a:np.ndarray):
        # obtain impulse response from filter coefficient
        w, h = freqz(b, a, fs=self.fs)

        # extract the amplitude and phase of impulse response
        amp = 20 * np.log10(abs(h))
        phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))

        return amp, phase, w


if __name__ == "__main__":
    # define input signal
    rate, signal = wavfile.read("example.wav")

    # apply pulsatile vocoder
    pulsatile_params = vocoder_params()
    pulsatile_params.sampling_frequency = rate
    vocoder = tone_vocoder(signal, pulsatile_params, "CIS")


    wavfile.write("result.wav", rate, vocoder.output)
