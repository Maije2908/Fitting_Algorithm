# Passive element type
#this has to be in a class namespace in order to use it in match/case szenarios -> this is annoying...very much so
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2

PROMINENCE_DEFAULT = 0.5
MAIN_RESONANCE_OFFSET = 50 #obsolete
MAX_ORDER = 15

# offset factor for the first resonance detected after the main peak (min_f = f0 * OFFSET_FACTOR)
# if the first resonance after the main peak is not fitted, consider setting this to a lower value
MIN_ZONE_OFFSET_FACTOR = 1.5

#multiplication factor for the parasitic element of the main resonance (C for inductor, L for capacitor)
MAIN_RES_PARASITIC_LOWER_BOUND = 0.5
MAIN_RES_PARASITIC_UPPER_BOUND = 2
#max/min values for the main resonance
MIN_R_FE = 10
MAX_R_FE = 1e9
MIN_R_ISO = 1e5
MAX_R_ISO = 1e12
R_ISO_VALUE = 1e15

#max/min values for the higher order circuits
RMAX = 1e5 #was 1e4
RMIN = 1e-3
LMIN = 1e-20
LOG_FACTOR = 1.04 #multiplication factor for the min bandwidth #OBSOLETE
BW_MIN_FACTOR = 1/1.5
BW_MAX_FACTOR = 1.5
MIN_CAP = 1e-20 #minimum capacitor
MAX_CAP_FACTOR = 1e5 #was 1e3
MIN_W_FACTOR = 0.99
MAX_W_FACTOR = 1.01

# factor to stretch the bandwidth of the last frequency zone (1 = no stretch)
BANDWIDTH_STRETCH_LAST_ZONE = 1

#number of samples to crop at the start of data
CROP_SAMPLES = 0







#parameters for the smoothing filter
SAVGOL_WIN_LENGTH = 52 #window length(samples) default:52
SAVGOL_POL_ORDER = 2 #polynomial order default:2

#offset for the calculation of the nominal parameters; useful if the phase data does not behave properly
#Note: changing this parameter can significantly improve goodness of fit (esp. for capacitors)
NOMINAL_VALUE_CALC_OFFSET = 3 #samples

MINIMUM_PRECISION = 1e-12 #if we encounter values that get singular, here is the threshold

DEFAULT_OFFSET_PEAK = 40 #samples; this specifies the default offset for a resonance peak if the 3dB point can't be found

#mode flags
class fcnmode:
    FIT: int = 1
    OUTPUT: int = 2

class multiple_fit:
    FULL_FIT = 1
    MAIN_RES_FIT = 2
