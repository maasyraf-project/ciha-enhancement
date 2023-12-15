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


class pulsatile_vocoder():
    '''
    The implementation of pulsatilve vocoder, as described in Brecker (2009)
    '''
    def __init__(self, input_signal: np.ndarray, params: tuple) -> np.ndarray:
        super(pulsatile_vocoder, self).__init__()

        # filterbank parameter
        self.order = params.filterbank_order
        self.fs = params.sampling_frequency

        # vocoder parameter
        self.nChan = params.number_of_channels
        self.cf = params.center_frequencies

        # process the signal
        self.output = self.analysis_filter(input_signal)

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

if __name__ == "__main__":
    # define input signal
    signal = np.random.normal(0, 0.5, 80000)

    # apply pulsatile vocoder
    pulsatile_params = vocoder_params()
    output_signal = pulsatile_vocoder(signal, pulsatile_params)
    print(np.shape(output_signal.output))
