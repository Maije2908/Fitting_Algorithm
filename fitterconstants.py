# Passive element type
#this has to be in a class namespace in order to use it in match/case szenarios -> this is annoying...very much so
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2

PROMINENCE_DEFAULT = 0.5
MAIN_RESONANCE_OFFSET = 50 #obsolete
MAX_ORDER = 15

#multiplication factor for the parasitic element of the main resonance (C for inductor, L for capacitor)
MAIN_RES_PARASITIC_LOWER_BOUND = 0.7
MAIN_RES_PARASITIC_UPPER_BOUND = 1.1
#max/min values for the higher order circuits
RMAX = 1e5 #was 1e4
RMIN = 1e-3
LMIN = 1e-20
LOG_FACTOR = 1.04 #multiplication factor for the min bandwidth #OBSOLETE
BW_MIN_FACTOR = 0.1
BW_MAX_FACTOR = 1.3
MIN_CAP = 1e-20 #minimm capacitor
MAX_CAP_FACTOR = 1e5 #was 1e3
MIN_W_FACTOR = 0.5
MAX_W_FACTOR = 1.5

DEFAULT_OFFSET_PEAK = 20 #samples; this specifies the default offset for a resonance peak if the 3dB point can't be found

#mode flags
class fcnmode:
    FIT: int = 1
    OUTPUT: int = 2
