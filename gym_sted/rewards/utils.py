
import numpy

from scipy import optimize

def gaussian(amplitude, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x) + 1e-3 # numerical stability
    width_y = float(width_y) + 1e-3 # numerical stability
    return lambda x,y: amplitude * numpy.exp(
        -1 * (((center_x - x) ** 2.0 / (2 * width_x ** 2.0)) + ((center_y - y) ** 2.0 / (2 * width_y ** 2.0)))
    )

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    X, Y = numpy.indices(data.shape)
    X = X - numpy.max(X) / 2
    Y = Y - numpy.max(Y) / 2
    errorfunction = lambda p: numpy.ravel(gaussian(*p)(X, Y) - data)
    p, success = optimize.leastsq(errorfunction, [data.max(), 0, 0, 1, 1], ftol=1e-3)
    return p


def validate_positions(guess_positions, acquired_signal, pixelsize, window_half_size=3, min_distance=2, threshold_rel=0.5):
    """
    Ensures that the detected guesses are actually gaussian-like structure with
    a valid size
    """
    padded_signal = numpy.pad(acquired_signal, window_half_size, constant_values=0)

    # then we try to fit a gaussian on every identified peak. If one of its sigma is > 200 nm, we reject the guess
    valid_guesses = []
    for row, col in guess_positions:
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
