# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import math
import matplotlib.pyplot as plt

class gammatone_filterbank():
    '''
    Implementation of computationally efficient of gammatone filterbank with equivalent rectangular bandwidth (ERB). Please refers to Hohmann (2002) for details of this implementation
    '''

    # global parameter


    # initiation function
    def __init__(self):
        super(gammatone_filterbank, self).__init__()

        # filter parameter


        # filter coefficient output

    def calculate_coefficient():

        # equation no. 14 in Hohmann (2002)
        a_gamma = (np.pi * math.factorial(2*self.order - 2)
            * pow(2, -(2*self.order - 2))
            / pow(math.factorial(self.order - 1),2))
        b = erb / a_gamma
        damping = np.exp(-2 * np.pi * b / frequency_sampling)


        # equation no. 1 in Hohmann (2002)
        analog_coefficient = damping * np.exp(1j * beta)
        coefficients = pow(number_of_sample, gamma - 1) * analog_coefficient

        return coefficients


class gammatone_filter():
    '''
    This class used to generate gammatone filterbank, based on this following reference
    [Hohmann 2002] : Frequency analysis and synthesis using a Gammatone filterbank, Acta Acustica (2002)

    '''

    # Global Parameter
    L = 24.7                    # see eq. (17) in [Hohmann 2002]
    Q = 9.265                   # see eq. (17) in [Hohmann 2002]

    def __init__(self, gamma_order, frequency_sampling, center_frequencies, bandwidth_factor):
        super(gammatone_filter, self).__init__()

        # filter parameter
        self.order = gamma_order
        self.fs = frequency_sampling
        self.cf = center_frequencies
        self.bf = bandwidth_factor
        self.norm_divisor = 1

        # filter coefficient
        self.filter, self.norm_factor = self.create_filter()

    def create_filter(self):
        '''
        This function used to create array of Gammatone Filterbank coefficient and normalization factor
        '''
        gamma_filter = np.zeros((len(self.cf),1), dtype="complex_")
        gamma_norm_factor = np.zeros((len(self.cf),1), dtype="complex_")
        for i in range(len(self.cf)):
            cf = self.cf[i]
            bf = self.bf[i]

            gamma_coef = self.calculate_coefficient(cf, bf)
            norm_fac = 2 * pow((1 - np.abs(gamma_coef)), self.order)

            gamma_filter[i] = gamma_coef
            gamma_norm_factor[i] = norm_fac

        return gamma_filter, gamma_norm_factor

    def calculate_coefficient(self, cf_chan, bf_chan):
        '''
        this function used to calculate coefficient on each channel of Gammatone Filterbank
        '''
        erb_aud = self.L + (cf_chan / self.Q) * bf_chan             # see eq. (13) in [Hohmann 2002]

        a_gamma = (np.pi * math.factorial(2*self.order - 2)         # see eq. (14), line 3 in [Hohmann 2002]
            * pow(2, -(2*self.order - 2))
            / pow(math.factorial(self.order - 1),2))

        b = erb_aud / a_gamma                                       # see eq. (14), line 2 in [Hohmann 2002]

        lambda_erb = np.exp(-2 * np.pi * b / self.fs)               # see eq. (14), line 1 in [Hohmann 2002]

        beta = 2 * np.pi * cf_chan / self.fs                        # see eq. (10) in [Hohmann 2002]

        coef = lambda_erb * np.exp(1j * beta)                       # see eq. (1), line 2 in [Hohmann 2002]

        return coef

if __name__ == "__main__":
    gamma_filt = gammatone_filter(4, 48000, [100, 200, 300], [1, 1, 1])