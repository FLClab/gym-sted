
"""This module contains classes that implement several objectives to optimize.
One can define a new objective by inheriting abstract class :class:`Objective`.
"""

from abc import ABC, abstractmethod

import numpy
import itertools
import warnings

from scipy.ndimage import gaussian_filter
from scipy import optimize
from skimage import measure, filters, morphology
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error

import metrics

from .utils import validate_positions

class Objective(ABC):
    """Abstract class to implement an objective to optimize. When inheriting this class,
    one needs to define an attribute `label` to be used for figure labels, and a
    function :func:`evaluate` to be called during optimization.
    """
    @abstractmethod
    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """Compute the value of the objective given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        raise NotImplementedError

    def mirror_ticks(self, ticks):
        """Tick values to override the true *tick* values for easier plot understanding.

        :param ticks: Ticks to replace.

        :returns: New ticks or None to keep the same.
        """
        return None


class Signal_Ratio(Objective):
    """Objective corresponding to the signal to noise ratio (SNR) defined by

    .. math::
        \\text{SNR} = \\frac{\\text{STED}_{\\text{fg}}^{75} - \overline{\\text{STED}_{\\text{fg}}}}{\\text{Confocal1}_{\\text{fg}}^{75}}

    where :math:`\\text{image}^q` and :math:`\\overline{\\text{image}}` respectively
    denote the :math:`q`-th percentile signal on an image and the average signal
    on an image, and :math:`\\text{STED}_{\\text{fg}}`, :math:`\\text{Confocal1}_{\\text{fg}}`, and
    :math:`\\text{Confocal2}_{\\text{fg}}` respectively refer to the foreground of the STED image
    and confocal images acquired before and after.

    :param float percentile: :math:`q`-th percentile in :math:`[0,100]`.
    """
    def __init__(self, percentile):
        self.label = "Signal Ratio"
        self.select_optimal = numpy.argmax
        self.percentile = percentile

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        """Compute the signal to noise ratio (SNR) given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: :math:`0` if no STED foreground, None if :math:`\\text{SNR} < 0` (error), or
                  SNR value otherwise.

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
    def __init__(self):
        self.label = "Bleach"
        self.select_optimal = numpy.argmin

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        signal_i = numpy.mean(confocal_init[confocal_fg])
        signal_e = numpy.mean(confocal_end[confocal_fg])
        bleach = (signal_i - signal_e) / signal_i
        return bleach

class Resolution(Objective):
    def __init__(self, pixelsize, res_cap=250):
        self.label = "Resolution (nm)"
        self.select_optimal = numpy.argmin
        self.pixelsize = pixelsize
#            self.kwargs = kwargs
        self.res_cap=res_cap

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.decorrelation(image=sted_stack[0])*self.pixelsize/1e-9
        if res > self.res_cap:
            res = self.res_cap
        return res

    def decorrelation(self, image):
        """
        Implements the decorrelation algorith from [...]

        :param image: A `numpy.ndarray` of the image

        :returns : A `float` of the resolution of the image
        """
        Nr = 50
        Ng = 10
        w = 20
        find_max_tol = 0.0005 #In the suppl info of the article it was 0.001...

        # Edge apodization
        edge_apod = (numpy.sin(numpy.linspace(-numpy.pi/2, numpy.pi/2, w))+1)/2
    #    Wx, Wy = numpy.meshgrid(*(numpy.concatenate([edge_apod, numpy.ones(l-2*w), numpy.flip(edge_apod)]) for l in image.shape)) #BEFORE
        Wx, Wy = numpy.meshgrid(*(numpy.concatenate([edge_apod, numpy.ones(l-2*w), numpy.flip(edge_apod)]) for l in numpy.flip(image.shape))) #AFTER
        W = Wx*Wy
        mean = image.mean()
        image = W*(image-mean) + mean


        # Do those two has any use ?
        image = image.astype('float32')
        image = image[:image.shape[0]-1+image.shape[0]%2,:image.shape[1]-1+image.shape[1]%2] #Odd array sizes

    #    X, Y = numpy.meshgrid(*(numpy.linspace(-1, 1, l) for l in image.shape)) #J'ai un ordre différent de dans le code matlab #BEFORE
        X, Y = numpy.meshgrid(*(numpy.linspace(-1, 1, l) for l in numpy.flip(image.shape))) #AFTER
        R = numpy.sqrt(X**2 + Y**2)


        Ik = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(image))) #Pourquoi utiliser fftshift 2 fois???
        Ik = Ik*(R<1) # ??? That was not in the "manual" ...

        with numpy.errstate(divide='ignore', invalid='ignore'):
            Ikn =  Ik/numpy.abs(Ik)
        Ikn[numpy.isnan(Ikn)] = 0 #Nécessaire?
        Ikn[numpy.isinf(Ikn)] = 0 #Nécessaire?


        def decorr(Ik, Ikn, R, r, highpass_sigma=None):

            Nr = len(r)
            if highpass_sigma:
               Ik =  Ik*(1 - numpy.exp(-2*highpass_sigma**2*R**2))

            d=[]
            denom2 = numpy.sum(numpy.abs(Ik)**2)
            Ikn_conj = numpy.conj(Ikn)
            Ikn_abs_exp2 = numpy.abs(Ikn)**2

            for i in range(Nr):
        #         Mr = R<r[i]
        #         I1 = Ik[Mr]
        #         I2_conj = Ikn_conj[Mr]
        #         I2_abs_exp2 = Ikn_abs_exp2[Mr]
        #         d.append(
        #         numpy.real(I1*I2_conj).sum()/numpy.sqrt(numpy.sum(I2_abs_exp2)*denom2)
        #         )
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
            d = numpy.floor(1000*d)/1000 # WHY ???
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

        sigmas = numpy.exp(numpy.arange(Ng+1)/Ng*(numpy.log(2/r0)-numpy.log(0.15)) + numpy.log(0.15)) #Not like in the code, Ng+1 values???
        gMax = 2/r0

        if gMax==numpy.inf: gMax = max(image.shape[0],image.shape[1])/2
        sigmas = numpy.array([image.shape[0]/4] + list(numpy.exp(numpy.linspace(numpy.log(gMax),numpy.log(0.15),Ng))))

        idxs = numpy.array([])
    #    As = numpy.array([A0]) #BEFORE
    #    rs = numpy.array([r0]) #BEFORE
        As = numpy.array([A0]) #AFTER
        rs = numpy.array([r0]) #AFTER

        for i, sig in enumerate(sigmas):
            d = decorr(Ik, Ikn, R, r, highpass_sigma=sig)
            idx, A = decorr_peak(d, find_max_tol) #Note that in the matlab code, the first element in the array is SNR0, so the length of the array A0 is 12 instead of 11 here
            idxs = numpy.append(idxs, idx)
            As = numpy.append(As, A)
            rs = numpy.append(rs, r[idx])

    #    print("line =", getframeinfo(currentframe()).lineno, "rs=", rs)
        max_freq_peak = rs.max()
        max_freq_peak_idx = numpy.where(rs == max_freq_peak)[0][-1]
    #    if r0>max_freq_peak: #BEFORE
        if max_freq_peak_idx==0: #AFTER
            ind1 = 0
        elif max_freq_peak_idx>=(len(sigmas)-1): #AFTER: == changed to >=
            ind1 = max_freq_peak_idx-2 #AFTER: -1 changed to -2
        else:
            ind1 = max_freq_peak_idx-1
        ind2 = ind1+1


        r1 = rs[max_freq_peak_idx] - (r[1]-r[0])
        r2 = rs[max_freq_peak_idx] + 0.4
        r_finetune = numpy.linspace(r1, min(r2,r[-1]), Nr)

    #    import pdb; pdb.set_trace()
    #    print("line =", getframeinfo(currentframe()).lineno, "r_finetune=", r_finetune)
        sigmas_fine_tune = numpy.exp(numpy.linspace(numpy.log(sigmas[ind1]), numpy.log(sigmas[ind2]) ,Ng))

        As=As[:-1] #AFTER (weird...)
        rs=rs[:-1] #AFTER (weird...)
        for i, sig in enumerate(sigmas_fine_tune):
            d = decorr(Ik, Ikn, R, r_finetune, highpass_sigma=sig)
            idx, A = decorr_peak(d, find_max_tol) #Note that in the matlab code, the first element in the array is SNR0, so the length of the array A0 is 12 instead of 11 here
            idxs = numpy.append(idxs, idx)
            As = numpy.append(As, A)
            rs = numpy.append(rs, r_finetune[idx])

    #    print("line =", getframeinfo(currentframe()).lineno, "rs=", rs)
        rs = numpy.append(rs, r0) #AFTER (nothing before)
        As = numpy.append(As, A0) #AFTER (nothing before)

        rs[As<0.05] = 0 # IN the matlab code, Ks set to zero too

        res = 2/numpy.max(rs)

        return res

class Squirrel(Objective):
    """
    Implements the `Squirrel` objective

    :param method: A `str` of the method used to optimize
    :param normalize: A `bool` wheter to normalize the images
    """
    def __init__(self, method="L-BFGS-B", normalize=False, use_foreground=False):

        self.label = "Squirrel"
        self.method = method
        self.bounds = (-1e+6, 1e+6), (-1e+6, 1e+6), (0, 1e+2)
        self.x0 = (1, 0, 1)
        self.normalize = normalize
        self.select_optimal = numpy.argmin
        self.use_foreground = use_foreground

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """
        Evaluates the objective

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        # Optimize
        if not numpy.any(sted_stack[0]):
            return 1.
#             return mean_squared_error(confocal_init[confocal_fg], sted_stack[0][confocal_fg], squared=False)
        # Optimize
        result = self.optimize(sted_stack[0], confocal_init)
        if self.use_foreground:
            return 1.0 - self.out_squirrel(result.x, sted_stack[0], confocal_init, confocal_fg=confocal_fg)
        else:
            return 1.0 - self.out_squirrel(result.x, sted_stack[0], confocal_init)

    def squirrel(self, x, *args, **kwargs):
        """
        Computes the reconstruction error between
        """
        alpha, beta, sigma = x
        super_resolution, reference = args
        confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
        convolved = self.convolve(super_resolution, alpha, beta, sigma)
        if self.normalize:
            reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
            convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
        error = mean_squared_error(reference[confocal_fg], convolved[confocal_fg], squared=True)
#         error = numpy.quantile(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]), [0.95]).item()
#         error = numpy.mean((reference[confocal_fg] - convolved[confocal_fg]))
        return error

    def out_squirrel(self, x, *args, **kwargs):
        """
        Computes the reconstruction error between
        """
        alpha, beta, sigma = x
        super_resolution, reference = args
        confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
        convolved = self.convolve(super_resolution, alpha, beta, sigma)
        if self.normalize:
            reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
            convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
#         error = numpy.mean(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]))
#         error = numpy.std(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]))
        _, S = structural_similarity(reference, convolved, full=True)
        error = numpy.mean(S[confocal_fg])
        return error

    def optimize(self, super_resolution, reference):
        """
        Optimizes the SQUIRREL parameters

        :param super_resolution: A `numpy.ndarray` of the super-resolution image
        :param reference: A `numpy.ndarray` of the reference image

        :returns : An `OptimizedResult`
        """
        result = optimize.minimize(
            self.squirrel, self.x0, args=(super_resolution, reference),
            method=self.method, bounds=(self.bounds)
        )
        return result

    def convolve(self, img, alpha, beta, sigma):
        """
        Convolves an image with the given parameters
        """
        return gaussian_filter(img * alpha + beta, sigma=sigma)

class NumberNanodomains(Objective):
    def __init__(self, threshold=2):
        # Do I need to inherit from the Objective class? this reward objective seems different from the others
        self.label = "NbNanodomains"
        self.select_optimal = None   # not sure what to put here ?????

        self.threshold = threshold

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        synapse = kwargs.get("synapse")

        filtered = filters.gaussian(sted_stack, sigma=0.5)
        threshold = numpy.quantile(filtered, 0.99)
        mask = filtered > threshold
        mask = morphology.remove_small_objects(mask, min_size=4, connectivity=2)
        labels = measure.label(mask)

        guess_coords = peak_local_max(filtered, min_distance=self.threshold, labels=labels, exclude_border=False)
        guess_coords = validate_positions(guess_coords, sted_stack, (synapse.datamap_pixelsize_nm) * 1e-9)

        gt_coords = numpy.asarray(synapse.nanodomains_coords)
        # guess_coords = peak_local_max(sted_stack, min_distance=self.threshold, threshold_rel=0.5)

        detector = metrics.CentroidDetectionError(gt_coords, guess_coords, self.threshold, algorithm="hungarian")
        f1_score = detector.f1_score
        return f1_score
