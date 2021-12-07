
"""This module contains classes that implement several objectives to optimize.
One can define a new objective by inheriting abstract class :class:`Objective`.
"""

from abc import ABC, abstractmethod

import numpy
import itertools
import warnings

from skimage.feature import peak_local_max

from . import metrics

class Objective(ABC):
    """
    Abstract class to implement an objective to optimize. When inheriting this class,
    one needs to define an attribute `label` to be used for figure labels, and a
    function :func:`evaluate` to be called during optimization.
    """
    @abstractmethod
    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """
        Compute the value of the objective given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        raise NotImplementedError

class Signal_Ratio(Objective):
    """
    Objective corresponding to the signal to noise ratio (SNR) defined by

    :param float percentile: q-th percentile in [0,100]
    """
    def __init__(self, percentile):
        self.label = "Signal Ratio"
        self.select_optimal = numpy.argmax
        self.percentile = percentile

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """
        Compute the signal to noise ratio (SNR) given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: A `float` of SNR value
        """
        if numpy.any(sted_fg):
            foreground = numpy.percentile(sted_stack[0][sted_fg], self.percentile)
            background = numpy.mean(sted_stack[0][numpy.invert(sted_fg)])
            ratio = (foreground - background) / numpy.percentile(confocal_init[confocal_fg], self.percentile)
            if ratio < 0:
                return None
            else:
                return ratio
        else:
            return 0

class Bleach(Objective):
    """
    Objective corresponding to the photobleaching of the acquired STED image
    """
    def __init__(self):
        self.label = "Bleach"
        self.select_optimal = numpy.argmin

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """Compute the photobleaching given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: A `float` of the photobleaching
        """
        signal_i = numpy.mean(confocal_init[confocal_fg])
        signal_e = numpy.mean(confocal_end[confocal_fg])
        bleach = (signal_i - signal_e) / signal_i
        return bleach

class Resolution(Objective):
    """
    Objective corresponding to the resolution of the acquired STED image

    The `max_resolution` is used in cases where the decorrelation algorithm cannot
    converge to a specifc resolution. In this work a value of 250nm is used which
    is typically the maximal resolution obtainable using confocal imaging.

    :param pixelsize: A `float` of the pixel size in m
    :param max_resolution: A `float` of the maximal resolution in nm that is achievable
    """
    def __init__(self, pixelsize, max_resolution=250):
        self.label = "Resolution (nm)"
        self.select_optimal = numpy.argmin
        self.pixelsize = pixelsize
        self.max_resolution=max_resolution

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """
        Compute the resolution given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: A `float` of the resolution of the image
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolution = self.decorrelation(image=sted_stack[0])*self.pixelsize/1e-9
        if resolution > self.max_resolution:
            resolution = self.max_resolution
        return resolution

    def decorrelation(self, image):
        """
        Implements the decorrelation algorithm from [1]

        [1] Descloux, A., Grußmayer, K. S. & Radenovic, A. Parameter-free image
            resolution estimation based on decorrelation analysis. Nature Methods
            16, 918–924 (2019).

        :param image: A `numpy.ndarray` of the image

        :returns : A `float` of the resolution of the image
        """
        Nr = 50
        Ng = 10
        w = 20
        find_max_tol = 0.0005

        edge_apod = (numpy.sin(numpy.linspace(-numpy.pi/2, numpy.pi/2, w))+1)/2
        Wx, Wy = numpy.meshgrid(*(numpy.concatenate([edge_apod, numpy.ones(l-2*w), numpy.flip(edge_apod)]) for l in numpy.flip(image.shape)))
        W = Wx*Wy
        mean = image.mean()
        image = W*(image-mean) + mean

        image = image.astype('float32')
        image = image[:image.shape[0]-1+image.shape[0]%2,:image.shape[1]-1+image.shape[1]%2]

        X, Y = numpy.meshgrid(*(numpy.linspace(-1, 1, l) for l in numpy.flip(image.shape)))
        R = numpy.sqrt(X**2 + Y**2)

        Ik = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(image)))
        Ik = Ik*(R<1)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            Ikn =  Ik/numpy.abs(Ik)
        Ikn[numpy.isnan(Ikn)] = 0
        Ikn[numpy.isinf(Ikn)] = 0

        def decorr(Ik, Ikn, R, r, highpass_sigma=None):

            Nr = len(r)
            if highpass_sigma:
               Ik =  Ik*(1 - numpy.exp(-2*highpass_sigma**2*R**2))

            d=[]
            denom2 = numpy.sum(numpy.abs(Ik)**2)
            Ikn_conj = numpy.conj(Ikn)
            Ikn_abs_exp2 = numpy.abs(Ikn)**2

            for i in range(Nr):
                if i==0:
                    Mr = R<r[i]
                    I1 = Ik[Mr]
                    I2_conj = Ikn_conj[Mr]
                    I2_abs_exp2 = Ikn_abs_exp2[Mr]
                    nom = numpy.real(I1*I2_conj).sum()
                    denom1 = numpy.sum(I2_abs_exp2)
                else:
                    Msk = numpy.logical_and(R<r[i], R>=r[i-1])
                    I1 = Ik[Msk]
                    I2_conj = Ikn_conj[Msk]
                    I2_abs_exp2 = Ikn_abs_exp2[Msk]
                    nom = nom + numpy.real(I1*I2_conj).sum()
                    denom1 = denom1 + numpy.sum(I2_abs_exp2)
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    d.append(
                        nom/numpy.sqrt(denom1*denom2)
                    )
            d=numpy.array(d)
            d = numpy.floor(1000*d)/1000
            d[numpy.isnan(d)] = 0
            return d

        r = numpy.linspace(0,1,Nr)
        d = decorr(Ik, Ikn, R, r)

        def decorr_peak(d, find_max_tol):
            idx = numpy.argmax(d)
            A = d[idx]
            while len(d) > 1:
                if ((A-numpy.min(d[idx:])) >= find_max_tol) or (idx==0):
                    break
                else:
                    d = d[:-1]
                    idx = numpy.argmax(d)
                    A = d[idx]
            return idx, A

        idx_0, A0 = decorr_peak(d, find_max_tol)
        r0 = r[idx_0]

        sigmas = numpy.exp(numpy.arange(Ng+1)/Ng*(numpy.log(2/r0)-numpy.log(0.15)) + numpy.log(0.15))
        gMax = 2/r0

        if gMax==numpy.inf: gMax = max(image.shape[0],image.shape[1])/2
        sigmas = numpy.array([image.shape[0]/4] + list(numpy.exp(numpy.linspace(numpy.log(gMax),numpy.log(0.15),Ng))))

        idxs = numpy.array([])
        As = numpy.array([A0])
        rs = numpy.array([r0])

        for i, sig in enumerate(sigmas):
            d = decorr(Ik, Ikn, R, r, highpass_sigma=sig)
            idx, A = decorr_peak(d, find_max_tol)
            idxs = numpy.append(idxs, idx)
            As = numpy.append(As, A)
            rs = numpy.append(rs, r[idx])
        max_freq_peak = rs.max()
        max_freq_peak_idx = numpy.where(rs == max_freq_peak)[0][-1]
        if max_freq_peak_idx==0:
            ind1 = 0
        elif max_freq_peak_idx>=(len(sigmas)-1):
            ind1 = max_freq_peak_idx-2
        else:
            ind1 = max_freq_peak_idx-1
        ind2 = ind1+1

        r1 = rs[max_freq_peak_idx] - (r[1]-r[0])
        r2 = rs[max_freq_peak_idx] + 0.4
        r_finetune = numpy.linspace(r1, min(r2,r[-1]), Nr)

        sigmas_fine_tune = numpy.exp(numpy.linspace(numpy.log(sigmas[ind1]), numpy.log(sigmas[ind2]) ,Ng))

        As=As[:-1]
        rs=rs[:-1]
        for i, sig in enumerate(sigmas_fine_tune):
            d = decorr(Ik, Ikn, R, r_finetune, highpass_sigma=sig)
            idx, A = decorr_peak(d, find_max_tol)
            idxs = numpy.append(idxs, idx)
            As = numpy.append(As, A)
            rs = numpy.append(rs, r_finetune[idx])

        rs = numpy.append(rs, r0)
        As = numpy.append(As, A0)

        rs[As<0.05] = 0

        res = 2/numpy.max(rs)

        return res

class NumberNanodomains(Objective):
    """
    Objective corresponding to the number of nanodomains detected

    :param threshold: An `int` of the threshold to associate truth and predicted
                      nanodomains
    """
    def __init__(self, threshold=2):
        self.label = "NbNanodomains"
        self.select_optimal = numpy.argmax
        self.threshold = threshold

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """
        Compute the F1-score given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: A `float` of the F1-score
        """
        synapse = kwargs.get("synapse")

        gt_coords = numpy.asarray(synapse.nanodomains_coords)
        guess_coords = peak_local_max(sted_stack, min_distance=self.threshold, threshold_rel=0.5)

        detector = metrics.CentroidDetectionError(gt_coords, guess_coords, self.threshold, algorithm="hungarian")
        f1_score = detector.f1_score
        return f1_score
