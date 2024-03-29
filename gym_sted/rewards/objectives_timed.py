
"""This module contains classes that implement several objectives to optimize.
One can define a new objective by inheriting abstract class :class:`Objective`.
***
This implementation is for the objectives of the 2nd version of the 2nd gym setting, in which datamaps evolve over time.
Some details, such as bleaching computation, differ due to this, which is why the implementation here differs from the
one in objectives.py
In this second version, SNR, Resolution and Bleach are still objectives, but the Nanodomains identification will be the
only objective used to compute rewards. The other objectives will be fed to the neural net as additional info
(or something like that idk)
***
"""

import numpy
import warnings
import metrics

from abc import ABC, abstractmethod
from skimage.feature import peak_local_max
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*numpy.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = numpy.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = numpy.sqrt(numpy.abs((numpy.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = numpy.sqrt(numpy.abs((numpy.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: numpy.ravel(gaussian(*p)(*numpy.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def find_nanodomains(acquired_signal, pixelsize, window_half_size=3, min_distance=2, threshold_rel=0.5):
    # first step is to find peaks in the image
    guess_positions = peak_local_max(acquired_signal, min_distance=min_distance, threshold_rel=threshold_rel)

    padded_signal = numpy.pad(acquired_signal, window_half_size, constant_values=0)

    # then we try to fit a gaussian on every identified peak. If one of its sigma is > 200 nm, we reject the guess
    valid_guesses = []
    for row, col in guess_positions:
        #TODO : what if the ND is on the edge of the image? I should test this :)
        data = padded_signal[(row + window_half_size) - window_half_size:
                             (row + window_half_size) + window_half_size + 1,
                             (col + window_half_size) - window_half_size:
                             (col + window_half_size) + window_half_size + 1]

        params = fitgaussian(data)
        # fwhm is approx 2.355 * sigma, sigma_x is params[3], sigma_y is params[4]
        fwhm_x, fwhm_y = 2.355 * params[3] * pixelsize, 2.355 * params[4] * pixelsize
        if fwhm_x > 200e-9 or fwhm_x < 40e-9 or fwhm_y > 200e-9 or fwhm_y < 40e-9:
            pass
        else:
            valid_guesses.append((row, col))
    valid_guesses = numpy.asarray(valid_guesses)

    return valid_guesses


class Objective(ABC):
    """Abstract class to implement an objective to optimize. When inheriting this class,
    one needs to define an attribute `label` to be used for figure labels, and a
    function :func:`evaluate` to be called during optimization.
    """
    @abstractmethod
    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        """Compute the value of the objective given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        :param n_molecs_init: The number of molecules in the BASE DATAMAP before the acquisition
        :param n_molecs_post: The number of molecules in the BASE DATAMAP after the acquisition
        :param temporal_datamap: The temporal_datamap object being acquired on, containing a synapse object with info on
                                 the number and positions of the nanodomains
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

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
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
            # foreground = numpy.percentile(sted_stack[0][sted_fg], self.percentile)
            # background = numpy.mean(sted_stack[0][numpy.invert(sted_fg)])
            foreground = numpy.percentile(sted_stack[sted_fg], self.percentile)
            background = numpy.mean(sted_stack[numpy.invert(sted_fg)])
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

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        # There is a RuntimeWarning: invalid value encountered in long_scalars that can happen
        # From my tests, I think it happens when n_molecs_init = n_molecs_post = 0
        # In that case, I think I want to set bleach to 100%, even if that action might not have bleached, since
        # everything is already bleached, which is not good :)
        if n_molecs_init == 0:
            bleach = 1.0
        else:
            bleach = (n_molecs_init - n_molecs_post) / n_molecs_init
        return bleach


class Resolution(Objective):
    def __init__(self, pixelsize, res_cap=250):
        self.label = "Resolution (nm)"
        self.select_optimal = numpy.argmin
        self.pixelsize = pixelsize
#            self.kwargs = kwargs
        self.res_cap = 250

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # res = self.decorrelation(image=sted_stack[0]) * self.pixelsize / 1e-9
            res = self.decorrelation(image=sted_stack) * self.pixelsize / 1e-9
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


class NumberNanodomains(Objective):
    def __init__(self):
        # Do I need to inherit from the Objective class? this reward objective seems different from the others
        self.label = "NbNanodomains"
        self.select_optimal = None   # not sure what to put here ?????

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        """
        Identify local maxima to 'guess' a number of nanodomains
        """
        gt_coords = numpy.asarray(temporal_datamap.synapse.nanodomains_coords)
        # guess_coords = peak_local_max(sted_stack, min_distance=threshold, threshold_rel=0.5)
        guess_coords = find_nanodomains(sted_stack, temporal_datamap.pixelsize)

        """
        problème potentiel :
        what if je start un acq pendant le flash, capable de bien résoudre les nanodomaines, mais genre au dernier
        pixel de l'acq (pour un exemple extreme, ça pourrait être plus tôt) ça update les flash et les nanodomaines ne
        sont plus actifs, ça me donnerait un reward de 0 malgré que les NDs sont biens résolus... hmm
        --> Will this ever happen tho? le flash ne décroit pas de full bien résolvable à off en un step, donc je pense 
            que le pire que ça va faire cest que je vais avoir un rwrd de 0 à place de genre 0.1 or something
        --> Quand même important à garder en tête, pourrais peut-être causer des comportements indésirés, mais pour 
            l'instant je vais garder cette implem
        """
        # if the nanodomains are not active, there is no ground truth position, so reward is 0
        if temporal_datamap.nanodomains_active_currently:
            detector = metrics.CentroidDetectionError(gt_coords, guess_coords, threshold, algorithm="hungarian")
            f1_score = detector.f1_score
        else:
            f1_score = 0

        return f1_score
