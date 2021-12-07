
import numpy

from gym_sted.rewards import objectives

P_STED = 35.0e-3
P_EX = 25.0e-6
PDT = 10.0e-6

LASER_EX = {"lambda_" : 488e-9}
LASER_STED = {"lambda_" : 575e-9, "zero_residual" : 0.01}
DETECTOR = {"noise" : True}
OBJECTIVE = {}
FLUO = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.08e-21,
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 3.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "phy_react": {
        488: 0.008e-5,
        575: 0.008e-8
    },
    "k_isc": 0.48e+6
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}
obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
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
