
import numpy

from .rewards import objectives

P_STED = 150.0e-3
P_EX = 10.0e-6
PDT = 10.0e-6

LASER_EX = {"lambda_" : 635e-9}
LASER_STED = {"lambda_" : 750e-9, "zero_residual" : 0.01, "anti_stoke": False}
DETECTOR = {"noise" : True, "background" : 0.5 / PDT}
OBJECTIVE = {
    "transmission" : {488: 0.84, 535: 0.85, 550: 0.86, 585: 0.85, 575: 0.85, 635: 0.84, 690: 0.82, 750: 0.77, 775: 0.75}
}
FLUO = {
    "lambda_": 6.9e-7,
    "qy": 0.65,
    "sigma_abs": {
        635: 3.2e-21,
        750: 3.5e-25
    },
    "sigma_ste": {
        750: 3.0e-22
    },
    "tau": 3.5e-9,
    "tau_vib": 1e-12,
    "tau_tri": 0.0000012,
    "k0": 0,
    "k1": 2.9e-16,
    "b": 1.66,
    "triplet_dynamics_frac": 0
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 10.0e-6},
    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
}

# Define the objectives and regressors here
obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9, res_cap=225),
    "Squirrel" : objectives.Squirrel(normalize=True, use_foreground=True),
    "NbNanodomains" : objectives.NumberNanodomains()
}

bounds_dict = {
    "SNR" : {"low" : 0.20, "high" : numpy.inf},
    "Bleach" : {"low" : -numpy.inf, "high" : 0.5},
    "Resolution" : {"low" : 0, "high" : 100},
    "NbNanodomains" : {"low" : 0, "high" : numpy.inf}
}

scales_dict = {
    "SNR" : {"low" : 0, "high" : 1},
    "Bleach" : {"low" : 0, "high" : 1},
    "Resolution" : {"low" : 40, "high" : 180},
    "NbNanodomains" : {"low" : 0, "high" : 1}
}

fluorescence_criterions = {
    "bleach" : {
        "p_ex" : [1.0e-6, 10.0e-6],
        "p_sted" : [100e-3, 200e-3],
        "pdt" : [1.0e-6, 10.0e-6],
        "target" : [0.1, 0.9]
    },
    "signal" : { # Avoids breaking the microscope with saturation of detector
        "p_ex" : 10.0e-6,
        "p_sted" : 0.,
        "pdt" : 10.0e-6,
        "target" : [10., 200.]
    },
}
