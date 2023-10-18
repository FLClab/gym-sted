
import numpy

from .rewards import objectives

DATAMAP_PATH = [
    "./data/datamap/actin",
    "./data/datamap/psd95",
    "./data/datamap/lifeact",
    "./data/datamap/PSD95-Bassoon",                
    "./data/datamap/tubulin",
    "./data/datamap/camkii"
]

P_STED = 150.0e-3
P_EX = 10.0e-6
PDT = 10.0e-6

LASER_EX = {"lambda_" : 635e-9}
LASER_STED = {"lambda_" : 750e-9, "zero_residual" : 0.01, "anti_stoke": False}
DETECTOR = {"noise" : True, "background" : 0.5 / (3 * PDT)}
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

FLUO_x10 = {
    # wavelength of the fluorophores, doesn't seem to change much if we stay close-ish (very large) to the beam
    # wavelengths, but going too far (i.e. 100e-9) renders the fluorophores much less "potent"
    "lambda_": 535e-9,
    "qy": 0.6,   # increasing increases the number of photons, decreasing decreases it
    "sigma_abs": {
        # beam - molecule interaction for excitation
        # modifying 575 (STED) doesn't seem to have much impact?
        # increasing 488 (EXC) increases number of photns, while decreasing it decreases the number of photons
        488: 0.03e-22,
        575: 0.02e-21
    },
    "sigma_ste": {
        # beam - molecule interaction for STED
        # decreasing the value decreases the STED effect, making the img more confocal for same STED power
        # increasing the value increases STED effect without increasing photobleaching cause by STED
        575: 2.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_vib": 1.0e-12,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_tri": 1.2e-6,   # decreasing decreases photobleaching, increasing increases photobleaching ?
    "phy_react": {
        488: 0.0008e-6,   # photobleaching caused by exc beam, lower = less photobleaching
        575: 0.00185e-8    # photobleaching cuased by sted beam, lower = less photobleaching
    },
    "k_isc": 0.48e+6,
}

FLUO_x100 = {
    # wavelength of the fluorophores, doesn't seem to change much if we stay close-ish (very large) to the beam
    # wavelengths, but going too far (i.e. 100e-9) renders the fluorophores much less "potent"
    "lambda_": 535e-9,
    "qy": 0.6,   # increasing increases the number of photons, decreasing decreases it
    "sigma_abs": {
        # beam - molecule interaction for excitation
        # modifying 575 (STED) doesn't seem to have much impact?
        # increasing 488 (EXC) increases number of photns, while decreasing it decreases the number of photons
        488: 0.0275e-23,
        575: 0.02e-21
    },
    "sigma_ste": {
        # beam - molecule interaction for STED
        # decreasing the value decreases the STED effect, making the img more confocal for same STED power
        # increasing the value increases STED effect without increasing photobleaching cause by STED
        575: 2.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_vib": 1.0e-12,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_tri": 1.2e-6,   # decreasing decreases photobleaching, increasing increases photobleaching ?
    "phy_react": {
        488: 0.0008e-6,   # photobleaching caused by exc beam, lower = less photobleaching
        575: 0.00185e-8   # photobleaching cuased by sted beam, lower = less photobleaching
    },
    "k_isc": 0.48e+6,
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 20.0e-6},
    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
}
abberior_action_spaces = {
    "p_sted" : {"low" : 0., "high" : 80.},
    "p_ex" : {"low" : 0., "high" : 18.},
    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
}

# Define the objectives and regressors here
obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9, res_cap=300),
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

# fluorescence_criterions = {
#     "bleach" : {
#         "p_ex" : [1.0e-6, 10.0e-6],
#         "p_sted" : [100e-3, 200e-3],
#         "pdt" : [1.0e-6, 10.0e-6],
#         "target" : [0.1, 0.9]
#     },
#     "signal" : { # Avoids breaking the microscope with saturation of detector
#         "p_ex" : 10.0e-6,
#         "p_sted" : 0.,
#         "pdt" : 10.0e-6,
#         "target" : [10., 200.]
#     },
# }
fluorescence_criterions = {
    "bleach" : {
        "p_ex" : [1.0e-6, 10.0e-6],
        "p_sted" : [100e-3, 200e-3],
        "pdt" : [5.0e-6, 75.0e-6],
        "target" : [0.1, 0.9]
    },
    "signal" : { # Avoids breaking the microscope with saturation of detector
        "p_ex" : [1.0e-6, 10.0e-6],
        "p_sted" : 0.,
        "pdt" : 10.0e-6,
        "target" : [10., 400.]
    },
}
