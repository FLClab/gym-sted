
import numpy, random
import json, os

from gym_sted.utils import BleachSampler

ROUTINES = {
    "high-signal_low-bleach" : {
        "bleach" : {
            "p_ex" : 10e-6,
            "p_sted" : 150e-3,
            "pdt" : 30.0e-6,
            "target" : 0.2
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 200.
        },
    },
    "high-signal_high-bleach" : {
        "bleach" : {
            "p_ex" : 2e-6,
            "p_sted" : 150e-3,
            "pdt" : 25.0e-6,
            "target" : 0.7
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 200.
        },
    },
    "low-signal_low-bleach" : {
        "bleach" : {
            "p_ex" : 10e-6,
            "p_sted" : 150e-3,
            "pdt" : 30.0e-6,
            "target" : 0.2
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 30.
        },
    },
    "low-signal_high-bleach" : {
        "bleach" : {
            "p_ex" : 2e-6,
            "p_sted" : 150e-3,
            "pdt" : 25.0e-6,
            "target" : 0.7
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 30.
        },
    }
}

if __name__ == "__main__":

    out = {}
    for key, routine in ROUTINES.items():
        random.seed(42)
        numpy.random.seed(42)

        sampler = BleachSampler("uniform", criterions=routine)
        parameters = sampler.sample()

        out[key] = parameters

    json.dump(out, open("routines.json", "w"), indent=4, sort_keys=True)
