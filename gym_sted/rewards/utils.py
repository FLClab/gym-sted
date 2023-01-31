
import numpy

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


def validate_positions(guess_positions, acquired_signal, pixelsize, window_half_size=3, min_distance=2, threshold_rel=0.5):
    """
    Ensures that the detected guesses are actually gaussian-like structure with
    a valid size
    """
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
