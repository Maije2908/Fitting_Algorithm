
# The fitter class shall take the task of fitting the data, as well as smoothing it and performing manipulations
# This is likely to become a rather long task, especially for CMCs and this class is therefore likely to be long
# I do not know yet what it will have to contain and how to best handle the data
# Most of this class will be based on Payer's program
#NOTE: THE CLASS IN THE FORM THAT IT IS NOW IS NOT ABLE TO MANAGE MULTIPLE FILES!!!!!




import numpy as np
import scipy
from scipy.signal import find_peaks
from lmfit import minimize, Parameters, Minimizer, report_fit
import matplotlib.pyplot as plt



class Fitter:

    def __init__(self):
        self.nominal_value = None
        self.parasitive_resistance = None
        self.prominence = None
        self.saturation = None
        self.files = None
        self.z21_data = None
        self.data_mag = None
        self.data_ang = None
        #using the Parameters() class from lmfit, might be necessary to make this an array when handling multiple files
        self.parameters = Parameters()

        self.frequency_zones = None

        #TODO: maybe change this; the f0 variable is for testing puropses
        self.f0 = None


    ####################################################################################################################
    # Parsing Methods
    ####################################################################################################################

    #method to set the entry values of the specification
    def set_specification(self, pass_val, para_r, prom, sat, fit_type):
        self.prominence = prom
        self.saturation = sat

        if pass_val is None:
            self.calculate_nominal_value(fit_type)
        else:
            self.nominal_value = pass_val

        if para_r is None:
            self.calculate_nominal_Rs()
        else:
            self.parasitive_resistance = para_r



    #method to parse the files from the iohandler
    def set_files(self, files):
        self.files = files



    ####################################################################################################################
    # Pre-Processing Methods
    ####################################################################################################################

    def calc_series_thru(self, Z0):
        for file in self.files:
            self.z21_data = 2 * Z0 * ((1 - file.data.s[:, 1, 0]) / file.data.s[:, 1, 0])

    def calc_shunt_thru(self, Z0):
        for file in self.files:
            self.z21_data = (Z0 * file.data.s[:, 1, 0]) / (2 * (1 - file.data.s[:, 1, 0]))


    def smooth_data(self):
        # Use Savitzky-Golay filter for smoothing the input data, because in the region of the global minimum there is
        # oscillation. After filtering a global minimum can be found easier.
        sav_gol_mode = 'interp'
        self.data_mag = scipy.signal.savgol_filter(abs(self.z21_data), 51, 2, mode=sav_gol_mode)
        self.data_ang = scipy.signal.savgol_filter(np.angle(self.z21_data, deg=True), 51, 2, mode=sav_gol_mode)
        #limit the data to +/- 90°
        self.data_ang = np.clip(self.data_ang, -90, 90)

        return 0

        # calculates the nominal value from the inductive/capacitive measured data, can be used, if nominal value is not specified
        # returns the nominal value <- copied from payer's program
        # fit_type -> 1-> inductor / 2-> capacitor / 3-> cmc (doesn't work)

    def calculate_nominal_value(self, fit_type):
        offset = 10  # samples
        nominal_value = 0
        freq = self.files[0].data.f

        match fit_type:
            case 1: #INDUCTOR

                # find first point where the phase crosses 0
                index_angle_smaller_zero = np.argwhere(self.data_ang < 0)
                index_ang_zero_crossing = index_angle_smaller_zero[0][0] # this somehow has to be "double unwrapped"

                if max(self.data_ang[offset:index_ang_zero_crossing]) < 88:
                    raise Exception("Error: Inductive range not detected (max phase = {value}°).\n"
                                    "Please specify nominal inductance.".format(value=np.round(max(self.data_ang), 1)))
                for sample in range(offset, len(freq)):
                    if self.data_ang[sample] == max(self.data_ang[offset:index_ang_zero_crossing]):
                        self.nominal_value = self.data_mag[sample] / 2 / np.pi / freq[sample]
                        break

            case 2: #CAPACITOR

                # find first point where the phase crosses 0
                index_angle_larger_zero = np.argwhere(self.data_ang > 0)
                index_ang_zero_crossing = index_angle_larger_zero[0][0]  # this somehow has to be "double unwrapped"

                if min(self.data_ang[offset:index_ang_zero_crossing]) > -88:
                    raise Exception("Error: Capacitive range not detected (min phase = {value}°).\n"
                                    "Please specify nominal capacitance.".format(value=np.round(min(self.data_ang), 1)))

                test_values = []
                for sample in range(offset, len(freq)):
                    if self.data_ang[sample] == min(self.data_ang[offset:index_ang_zero_crossing]):
                        nominal_value = 1 / (2 * np.pi * freq[sample] * self.data_mag[sample])
                        self.nominal_value = nominal_value
                        test_values.append(self.nominal_value)
                        # break
                test_values_gradient = abs(np.gradient(test_values, 2))
                # it takes the first values instead of the "linear" range. need to fix this. possibly by taking the longest min gradient

                self.nominal_value = test_values[np.argmin(np.amin(test_values_gradient))]
                print(self.nominal_value)
            case 3:
                self.nominal_value = 0
            case _:
                self.nominal_value = 0


        return self.nominal_value

    def calculate_nominal_Rs(self):
        R_s_input = min(self.data_mag)
        #TODO: logging and error handling ( method could be called before the data is initialized)
        self.parasitive_resistance = R_s_input


    def get_main_resonance(self, fit_type):
        freq = self.files[0].data.f

        #set w0 to 0 in order to have feedback, if the method didn't work
        w0 = 0

        #TODO: maybe check fit_type variable to be 1/2/3?

        match fit_type:

            case 1: #INDUCTOR
                index_angle_smaller_zero = np.argwhere(self.data_ang < 0)
                index_ang_zero_crossing = index_angle_smaller_zero[0][0]
                continuity_check = index_angle_smaller_zero[10][0]

            case 2: #CAPACITOR
                index_angle_larger_zero = np.argwhere(self.data_ang > 0)
                index_ang_zero_crossing = index_angle_larger_zero[0][0]
                continuity_check = index_angle_larger_zero[10][0]

            case 3: #CMC
                sign = 1 #TODO: i dont know what value to take for CMCs and how to handle them in general

        if continuity_check:
            f0 = freq[index_ang_zero_crossing]
            w0 = f0 * 2 * np.pi
            self.f0 = f0
            print("f0: {f0}".format(f0=f0))

        if w0 == 0:
            raise Exception('\nSystem Log: ERROR: Main resonant frequency could not be determined.')

        #TODO: write found w0 to parameters


    def get_resonances(self):

        min_prominence = 0.05
        prominence_mag = 0.01
        R_s = self.parasitive_resistance
        freq = self.files[0].data.f

        #TODO: find_peaks tends to detect "too many" peaks i.e. overfits!!! (Long term problem)

        #find peaks of Magnitude Impedance curve (using scipy.signal.find_peaks)
        mag_maxima = find_peaks(self.data_mag, height=np.log10(R_s), prominence=prominence_mag)
        mag_minima = find_peaks(self.data_mag * -1, prominence=prominence_mag)
        #find peaks of Phase curve
        phase_maxima = find_peaks(self.data_ang, prominence=prominence_mag)
        phase_minima = find_peaks(self.data_ang * -1, prominence=prominence_mag)

        #map to frequency; TODO: we are using the file here, so if there are multiple files, need to change this
        f_mag_maxima = freq[mag_maxima[0]]
        f_mag_minima = freq[mag_minima[0]]

        f_phase_maxima = freq[phase_maxima[0]]
        f_phase_minima = freq[phase_minima[0]]

        min_zone_start = self.f0 * 2  # frequency buffer for first RLC circuit TODO: AD "find_peaks" this might do the trick
        ang_minima_pos = f_phase_minima[f_phase_minima > min_zone_start]
        ang_maxima_pos = f_phase_maxima[f_phase_maxima > min_zone_start]

        #plot commands to check peak values
        #markerson = mag_maxima[0]
        #plt.loglog(self.data_mag,'-bD', markevery=markerson)

        #loop to find frequency ranges, copied from payer
        number_zones = len(ang_minima_pos)
        f_zones_list = []
        for minimum in range(0, number_zones):
            f1 = ang_minima_pos[minimum]
            f3 = max(freq) * 5
            if minimum + 1 < number_zones:
                f3 = ang_minima_pos[minimum + 1]

            # find the maxima between two minima
            for maximum in range(len(ang_maxima_pos)):
                if f1 < ang_maxima_pos[maximum] < f3:
                    f_tuple = (f1, ang_maxima_pos[maximum], f3)
                    f_zones_list.append(f_tuple)
                    break  # corresponding f2 found
        try:
            if ang_minima_pos[-1] > ang_maxima_pos[-1]:
                f_tuple = (ang_minima_pos[-1], max(freq) * 3, f3)
                f_zones_list.append(f_tuple)
        # no minima or maxima present - not sure if this works correctly
        except Exception as e:
            if number_zones > 0:  # else base model
                f_tuple = (ang_minima_pos[-1], np.sqrt(max(freq) ** 2 * 3), max(freq) * 6)
                f_zones_list.append(f_tuple)
                print("Warning from frequency zones: {e}".format(e=e))
                pass

        self.frequency_zones = f_zones_list



