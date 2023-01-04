# Passive element type
#this has to be in a class namespace in order to use it in match/case szenarios
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2

PROMINENCE_DEFAULT = .5
MAIN_RESONANCE_OFFSET = 50 #obsolete
MAX_ORDER = 15
FREQ_UPPER_LIMIT = 200e6 #2e9 #3.9e8 #2e9 #500e6 #2e9

# offset factor for the first resonance detected after the main peak (min_f = f0 * OFFSET_FACTOR)
# if the first resonance after the main peak is not fitted, consider setting this to a lower value
MIN_ZONE_OFFSET_FACTOR = 2

#multiplication factor for the parasitic element of the main resonance (C for inductor, L for capacitor)
MAIN_RES_PARASITIC_LOWER_BOUND = 0.5
MAIN_RES_PARASITIC_UPPER_BOUND = 2
#max/min values for the main resonance
MIN_R_FE = 10
MAX_R_FE = 1e9
MIN_R_ISO = 1e5
MAX_R_ISO = 1e9
R_ISO_VALUE = 10e6


#max/min values for the higher order circuits
RMAX = 1e5 #was 1e4
RMIN = 1e-3
LMIN = 1e-20
LOG_FACTOR = 1.04 #multiplication factor for the min bandwidth #OBSOLETE

MIN_CAP = 1e-20 #minimum capacitor
MAX_CAP_FACTOR = 1e5 #was 1e3


MAX_W_FACTOR = 1.0001
MIN_W_FACTOR = 1/MAX_W_FACTOR
BW_MAX_FACTOR = 1.01
BW_MIN_FACTOR = 1/BW_MAX_FACTOR

# factor to stretch the bandwidth of the last frequency zone (1 = no stretch)
BANDWIDTH_STRETCH_LAST_ZONE = 1


#threshold for the calculation of the offset; necessary for small coils that have a lot of zero crossings at low frequencies
PHASE_OFFSET_THRESHOLD = 60 #60 #°
PHASE_OFFSET_THRESHOLD_CAPS = 20
#value for detection of the inductive/capacitive range; if phase is below this value, inductive/capacitive range will not be detected
PERMITTED_MIN_PHASE = 75 #75


#Decide wheter to full fit the higher order resonances (0= main res fit; 1= full fit)
FULL_FIT = 1


#parameters for the smoothing filter
SAVGOL_WIN_LENGTH = 52 #window length(samples) default:52
SAVGOL_POL_ORDER = 2 #polynomial order default:2


#multiplication factor for statistical evaluation of the nominal values; this value will be multiplied to the .50 quanti
#le of the slope and gives the max deviation of the .50 quantile
QUANTILE_MULTIPLICATION_FACTOR = 5

MINIMUM_PRECISION = 1e-12 #if we encounter values that get singular, here is the threshold

DEFAULT_OFFSET_PEAK = 40 #samples; this specifies the default offset for a resonance peak if the 3dB point can't be found


#mode flags
class fcnmode:
    FIT:        int = 1
    FIT_LOG:    int = 6
    OUTPUT:     int = 2
    ANGLE:      int = 3
    FIT_REAL:   int = 4
    FIT_IMAG:   int = 5

class multiple_fit:
    FULL_FIT = 1
    MAIN_RES_FIT = 2

class calc_method:
    SERIES = 1
    SHUNT = 2

class captype:
    GENERIC = 1
    MLCC = 2
    HIGH_C = 3

class capunits:
    FARADS:         float = 1
    MILLIFARADS:    float = 1e-3
    MICROFARADS:    float = 1e-6
    NANOFARADS:     float = 1e-9

class indunits:
    HENRYS:         float = 1
    MILLIHENRIES:   float = 1e-3
    MICROHENRIES:   float = 1e-6

class funits:
    HERTZ:          float = 1
    KILOHERTZ:      float = 1e3
    MEGAHERTZ:      float = 1e6

#determines whether to generate differnce plots or not
OUTPUT_DIFFPLOTS = 1

#Debug Plots
DEBUG_BW_MODEL = 0
DEBUG_BW_MODEL_VERBOSE = 0
DEBUG_FIT = 0
DEBUG_MESSAGES = 1
DEBUG_BW_DETECTION = 0
DEBUG_MULTIPLE_FITE_FIT = 0#1


#determines whether to show bode plots or only save them
SHOW_BODE_PLOTS = False


#logging
LOGGING_VERBOSE = 0

