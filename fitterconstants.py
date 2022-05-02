# Passive element type
#this has to be in a class namespace in order to use it in match/case szenarios -> this is annoying...very much so
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2

PROMINENCE_DEFAULT = 0.5
MAIN_RESONANCE_OFFSET = 50 #obsolete
MAX_ORDER = 15

#multiplication factor for the parasitic element of the main resonance (C for inductor, L for capacitor)
MAIN_RES_PARASITIC_LOWER_BOUND = 0.5
MAIN_RES_PARASITIC_UPPER_BOUND = 2
#max/min values for the main resonance
MIN_R_FE = 10
MAX_R_FE = 1e9
MIN_R_ISO = 1
MAX_R_ISO = 1e12

#max/min values for the higher order circuits
RMAX = 1e5 #was 1e4
RMIN = 1e-3
LMIN = 1e-20
LOG_FACTOR = 1.04 #multiplication factor for the min bandwidth #OBSOLETE
BW_MIN_FACTOR = 0.2
BW_MAX_FACTOR = 1.5
MIN_CAP = 1e-20 #minimm capacitor
MAX_CAP_FACTOR = 1e5 #was 1e3
MIN_W_FACTOR = 0.98
MAX_W_FACTOR = 1.02

#parameters for the smoothing filter
SAVGOL_WIN_LENGTH = 52 #window length(samples) default:52
SAVGOL_POL_ORDER = 2 #polynomial order default:2


MINIMUM_PRECISION = 1e-12 #if we encounter values that get singular, here is the threshold

DEFAULT_OFFSET_PEAK = 20 #samples; this specifies the default offset for a resonance peak if the 3dB point can't be found

#mode flags
class fcnmode:
    FIT: int = 1
    OUTPUT: int = 2
