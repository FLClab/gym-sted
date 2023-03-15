
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
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}
