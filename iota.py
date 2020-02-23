#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Mai 16, 2017"
__version__ = "v2.1"

"""
This module serves estimating different variants of the iota (i) measure.
"""

import warnings

# Import NumPy for the array object and fast numerics.
import numpy as np

flag_numba = True
try:
    import numba
except:
    flag_numba = False
    warnings.warn("Could not import numba! Computation might be very slow. Try to use weave instead by setting flag_weave=True on init.")

# Import scipy.weave for backward compatibility.
try:
    try:
        import weave
    except:
        from scipy import weave
except:
    warnings.warn("Could not import weave! This will cause an error if you'll set flag_weave=True!")

from scipy.stats import norm
from scipy.stats import percentileofscore

def numba_decorator(func):
    if flag_numba:
        return numba.jit(func)
    else:
        return func


class IOTA(object):
    # TODO: weighting for squared slope?
    # TODO: revise significance test

    def __init__(self, method="iota", weighting="uniform", significance=None, normed=False, flag_weave=False):

        self.weighting = weighting
        self.normed = normed
        self.sig = significance
        self.num_surrogates = 500
        self.method_name = method

        self.flag_weave = flag_weave

        if method == "iota":
            self.method = self._iota
        elif method == "iotar":
            self.method = self._iota_r
        elif method == "biiota":
            self.method = self._biiota
        elif method == "fancy_iota":
            self.method = self._fancy_iota
        else:
            raise ValueError("Method needs to be iota, iotar, biiota or fancy_iota!")

    def __str__(self):
        print "\n---"
        for attr in sorted(vars(self)):
            print attr, ':', getattr(self, attr)
        print "---"
        return ""

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    def similarity(self, x, y):
        """returns the iota-variant value for the direction x to y"""

        if self.normed:
            x, y = self.normalize(x), self.normalize(y)

        g = y[self.permute(x)]

        return self.method(g)


    def conditional_similarity(self, x, y, z):
        """returns partial iota for x to y given z for given variant"""


        if self.normed:
            x, y, z = self.normalize(x), self.normalize(y), self.normalize(z)
        px, py, pz = self.permute(x), self.permute(y), self.permute(z)

        if self.method_name == "fancy_iota":
            # use sperman's rank correlation
            iac = self.method(z[px])
            iab = self.method(y[px])
            icb = self.method(y[pz])
            return 1. * (iac - iab * icb) / np.sqrt((1 - iab**2) * (1 - icb**2))
        else:
            return np.abs(self.method(y[px[pz]]) - self.method(y[pz]))

    @numba_decorator
    def surrogate_distribution(self, *args):
        if len(args) == 3:
            func = self.conditional_similarity
        elif len(args) == 2:
            func = self.similarity
        else:
            raise ValueError("Please input two timeseries (or three for conditional) similarity testing.")

        dist = np.zeros(self.num_surrogates)
        for j in xrange(self.num_surrogates):
            for t in args:
                np.random.shuffle(t)
            dist[j] = func(*args)

        return dist

    @numba_decorator
    def significance_Hempel(self, estimate, *args):
        """
        returns significance from shuffled surrogate_distribution distribution.
        hypothesis accepted if i larger then mean iota of surrogates
        cf. Hempel et al., EPJ B (2013)

        This works both for the pairwise as well as the conditional variants of iota.
        """
        dist = self.surrogate_distribution(*args)

        return estimate > np.mean(dist)

    def significance_onesided(self, estimate, *args):
        """

        :param estimate:
        :param sig: must be in [0;100]
        :param args:
        :return:
        """

        dist = self.surrogate_distribution(*args)

        return 1. - percentileofscore(dist, estimate, 'weak') / 100.

    def significance_twosided(self, estimate, *args):
        """

        :param estimate:
        :param sig: must be in [0;100]
        :param args:
        :return:
        """
        dist = self.surrogate_distribution(*args)

        p = percentileofscore(dist, estimate, 'weak')

        if estimate < 0:
            return p / 100.
        elif estimate > 0:
            return 1. - p / 100.
        return 0.5

    ###############################################################################
    # ##                       PRIVATE FUNCTIONS                               ## #
    ###############################################################################

    @staticmethod
    def normalize(arr, typ = "gaussian"):
        if typ == "gaussian":
            #normalizes to gaussian
            return np.array([norm.ppf(percentileofscore(arr, arr[i], 'mean') / 100.) for i in xrange(len(arr))])
        else:
            # normalizes arr to the interval [0, 1]
            return 1. * (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))


    @staticmethod
    def permute(array):
        p = np.argsort(array, axis=0)
        return p

    def _biiota(self, g):
        i1 = self._iota(np.array(g))
        i2 = self._iota_r(np.array(g))
        return 1. * (i1 + i2) / 2.

    def _iota(self, g):
        return self.__iota_weave(g) if self.flag_weave else self.__iota_numba(g)

    def _iota_r(self, g):
        return self.__iota_r_weave(g) if self.flag_weave else self.__iota_r_numba(g)

    def _fancy_iota(self, g):
        return self.__fancy_iota_weave(g) if self.flag_weave else self.__fancy_iota_numba(g)

    @numba_decorator
    def __iota_numba(self, g):
        N = len(g)

        temp = 0.

        if self.weighting == "squared_slope":

            for i in range(0, N-2):
                for j in range(i+1, N-1):
                    if (g[j+1] - g[i]) * (g[i] - g[j]) > 0:
                        temp += (g[j+1] - g[j]) ** 2

            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))

        elif self.weighting == "uniform":

            for i in range(0, N-2):
                for j in range(i+1, N-1):
                    if (g[j+1] - g[i]) * (g[i] - g[j]) > 0:
                        temp += 1

            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))
        else:
            raise ValueError("Invalid weighting! Must be uniform or squared_slope.")

    def __iota_weave(self, g):
        N = len(g)

        if self.weighting == "squared_slope":

            code = \
                '''
            double temp = 0.;

            for(int i=0;i<N-2;i++){
                for(int j=i+1;j<N-1;j++){
                    if( (G1(j+1)-G1(i))*(G1(i)-G1(j)) > 0) {
                        temp = temp + (G1(j+1)-G1(j))*(G1(j+1)-G1(j));

                        //if((G1(j+1)-G1(j))*(G1(j+1)-G1(j)) > 0){
                        //    temp = temp + (G1(j+1)-G1(j))*(G1(j+1)-G1(j));
                        //    }
                        //else {
                        //    temp = temp + 1;
                        //    }
            }}}
            return_val = temp;
            '''
            temp = weave.inline(code, ['N', 'g'], compiler="gcc", headers=["<stdio.h>"])
            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))

        elif self.weighting == "uniform":
            code = \
                '''
            double temp =0;
            for(int i=0;i<N-2;i++){
                for(int j=i+1;j<N-1;j++){
                    if( (G1(j+1)-G1(i))*(G1(i)-G1(j)) > 0) {
                        temp = temp + 1.;
            }}}
            return_val = temp;
            '''
            temp = weave.inline(code, ['N', 'g'], compiler="gcc", headers=["<stdio.h>"])
            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))
        else:
            raise ValueError("Invalid weighting! Must be uniform or squared_slope.")

    @numba_decorator
    def __iota_r_numba(self, g):
        N = len(g)

        temp = 0.

        if self.weighting == "squared_slope":

            for i in range(0, N - 2):
                for j in range(i + 1, N - 1):
                    if (g[j - 1] - g[i]) * (g[i] - g[j]) > 0:
                        temp += (g[j + 1] - g[j]) ** 2

            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))

        elif self.weighting == "uniform":

            for i in range(0, N - 2):
                for j in range(i + 1, N - 1):
                    if (g[j - 1] - g[i]) * (g[i] - g[j]) > 0:
                        temp += 1

            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))
        else:
            raise ValueError("Invalid weighting! Must be uniform or squared_slope.")

    def __iota_r_weave(self, g):
        N = len(g)
        if self.weighting == 'squared_slope':

            code = \
                '''
            double temp = 0.;

            for(int i=2;i<N-1;i++){
                for(int j=1;j<i-1;j++){
                    if( (G1(j-1)-G1(i))*(G1(i)-G1(j)) > 0) {
                        temp = temp + (G1(j+1)-G1(j))*(G1(j+1)-G1(j));

                        //if((G1(j+1)-G1(j))*(G1(j+1)-G1(j)) > 0){
                        //    temp = temp + (G1(j+1)-G1(j))*(G1(j+1)-G1(j));
                        //    }
                        //else {
                        //    temp = temp + 1;
                        //    }
            }}}
            return_val = temp;
            '''
            temp = weave.inline(code, ['N', 'g'], compiler="gcc", headers=["<stdio.h>"])
            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))
        elif self.weighting == 'uniform':
            code = \
                '''
            double temp = 0.;

            for(int i=2;i<N-1;i++){
                for(int j=1;j<i-1;j++){
                    if( (G1(j-1)-G1(i))*(G1(i)-G1(j)) > 0) {
                        temp = temp + 1.;
            }}}
            return_val = temp;
            '''
            temp = weave.inline(code, ['N', 'g'], compiler="gcc", headers=["<stdio.h>"])
            return 1. - temp * 2.0 / ((N - 1.) * (N - 2.))
        else:
            raise ValueError("Invalid weighting! Must be uniform or squared_slope.")

    @numba_decorator
    def __fancy_iota_numba(self, g):
        N = len(g)

        increment = 0
        norm = 0
        for i in range(0, N-2):
            for j in range(i+1, N-1):
                increment += (g[j] - g[i]) * np.abs(g[j] - g[i])
                norm += (g[j] - g[i]) ** 2

        return 1. * increment / norm

    def __fancy_iota_weave(self, g):
        N = len(g)

        '''
        need to verify correct normalisation
        '''

        code = \
            '''
        double increments = 0.;
        double norm = 0.;

        for(int i=0;i<N-2;i++){
            for(int j=i+1;j<N-1;j++){
                increments = increments + (G1(j)-G1(i))*fabs(G1(j)-G1(i));
                norm = norm + (G1(j)-G1(i))*(G1(j)-G1(i));
            }
        }
        return_val = increments / norm;
        '''

        return weave.inline(code, ['N', 'g'], compiler="gcc", headers=["<stdio.h>","<math.h>"])



def tests():
    import itertools
    import xarray

    np.random.seed(0)

    T = 40
    times = np.linspace(0, T, 100)
    x = np.sin(2 * np.pi * times / T)
    y = x + (1 - 2 * np.random.random(len(x)))
    z = x + (1 - 2 * np.random.random(len(x)))

    data = xarray.Dataset({"x": ("time", x), "y": ("time", y), "z": ("time", z)}, coords={"time": times})
    #[data[k].plot() for k in data.data_vars.keys()]

    for method in ["iota", "biiota", "fancy_iota"]:

        for weighting in ["uniform"]:#, "squared_slope"]:

            iota = IOTA(method=method, weighting=weighting, normed=False)
            iota.num_surrogates = 1000

            if method == "fancy_iota":
                test = iota.significance_twosided
            else:
                test = iota.significance_onesided

            # pairwise
            for l in itertools.permutations(data.data_vars.keys(), 2):
                io = iota.similarity(data[l[0]].values, data[l[1]].values)
                print method, weighting, l, io, test(io, data[l[0]].values, data[l[1]].values)


            # conditional
            for l in itertools.permutations(data.data_vars.keys(), 3):
                io = iota.conditional_similarity(data[l[0]].values, data[l[1]].values, data[l[2]].values)
                print method, weighting, l, io, test(io, data[l[0]].values, data[l[1]].values, data[l[2]].values)

if __name__ == "__main__":
    # TODO: implement tests!
    #import doctest
    #doctest.testmod()
    tests()