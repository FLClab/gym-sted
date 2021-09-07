
P_STED = 2.5e-3
P_EX = 2.0e-6
PDT = 100.0e-6

LASER_EX = {"lambda_" : 488e-9}
LASER_STED = {"lambda_" : 575e-9, "zero_residual" : 0.01}
DETECTOR = {"noise" : True}
OBJECTIVE = {}
FLUO = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {488: 4.5e-21,   # was 1.15e-20
                  575: 6e-21},
    "sigma_ste": {560: 1.2e-20,
                  575: 3.0e-22,   # was 6.0e-21
                  580: 5.0e-21},
    "sigma_tri": 1e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 5e-6,
    "phy_react": {488: 0.25e-7,   # 1e-4
                  575: 25.0e-11},   # 1e-8
    "k_isc": 0.26e6
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 100.0e-3},
    "p_ex" : {"low" : 0., "high" : 5.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 75.0e-6},
}