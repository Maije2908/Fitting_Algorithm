
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
        self.prominence = 0
        self.saturation = 0
        self.files = None
        self.z21_data = None
        self.data_mag = None
        self.data_ang = None
        self.fit_type = None
        self.out = None
        #using the Parameters() class from lmfit, might be necessary to make this an array when handling multiple files
        self.parameters = Parameters()

        self.frequency_zones = None


        #TODO: maybe change this; the f0 variable is for testing puropses
        self.f0 = None
        self.max_order = 15 #TODO: this is hard-coded for testing purposes
        self.order = None


    ####################################################################################################################
    # Parsing Methods
    ####################################################################################################################

    #method to set the entry values of the specification
    def set_specification(self, pass_val, para_r, prom, sat, fit_type):

        self.fit_type = fit_type

        if pass_val is None:
            self.calculate_nominal_value()
        else:
            self.nominal_value = pass_val

        if para_r is None:
            self.calculate_nominal_Rs()
        else:
            self.parasitive_resistance = para_r

        if prom is None:
            self.prominence = 0
        else:
            self.prominence = prom

        if sat is None:
            self.saturation = 0
        else:
            self.saturation = sat



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

    def calculate_nominal_value(self):
        offset = 10  # samples
        nominal_value = 0
        freq = self.files[0].data.f

        match self.fit_type:
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

    def get_main_resonance(self):
        freq = self.files[0].data.f

        #set w0 to 0 in order to have feedback, if the method didn't work
        w0 = 0

        #TODO: maybe check fit_type variable to be 1/2/3?

        match self.fit_type:

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

        min_prominence_phase = 0.5
        prominence_mag = 0.01
        R_s = self.parasitive_resistance
        freq = self.files[0].data.f
        prominence_phase = max(min_prominence_phase, float(self.prominence))

        #TODO: find_peaks tends to detect "too many" peaks i.e. overfits!!! (Long term problem)

        #find peaks of Magnitude Impedance curve (using scipy.signal.find_peaks)
        mag_maxima = find_peaks(self.data_mag, height=np.log10(R_s), prominence=prominence_mag)
        mag_minima = find_peaks(self.data_mag * -1, prominence=prominence_mag)
        #find peaks of Phase curve
        phase_maxima = find_peaks(self.data_ang, prominence=prominence_phase)
        phase_minima = find_peaks(self.data_ang * -1, prominence=prominence_phase)

        #map to frequency; TODO: we are using the file here, so if there are multiple files, need to change this
        #TODO: why are we even calculating the magnitude maxima if they are never used???
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
        for num_minimum in range(0, number_zones):
            f1 = ang_minima_pos[num_minimum]
            f3 = max(freq) * 5
            if num_minimum + 1 < number_zones:
                f3 = ang_minima_pos[num_minimum + 1]

            # find the maxima between two minima
            for num_maximum in range(len(ang_maxima_pos)):
                if f1 < ang_maxima_pos[num_maximum] < f3:
                    f_tuple = (f1, ang_maxima_pos[num_maximum], f3)
                    f_zones_list.append(f_tuple)
                    break  # corresponding f2 found
        try:
            if ang_minima_pos[-1] > ang_maxima_pos[-1]:
                f_tuple = (ang_minima_pos[-1], max(freq) * 3, f3)
                f_zones_list.append(f_tuple)
        # no minima or maxima present - not sure if this works correctly TODO: me neither, but let's assume it works for the moment
        except Exception as e:
            if number_zones > 0:  # else base model
                f_tuple = (ang_minima_pos[-1], np.sqrt(max(freq) ** 2 * 3), max(freq) * 6)
                f_zones_list.append(f_tuple)
                print("Warning from frequency zones: {e}".format(e=e))
                pass

        self.frequency_zones = f_zones_list

    def create_nominal_parameters(self):

        self.parameters.add('R_s', value=self.parasitive_resistance, min=1e-20, vary=False)

        #max/min values for the isolation/iron resistance
        min_R_Fe    = 10
        max_R_Fe    = 1e9
        min_R_iso   = 1
        max_R_iso   = 1e12

        match self.fit_type:
            case 1:
                #calculate "perfect" capacitor for this resonance
                cap_ideal = 1 / (self.nominal_value * ((self.f0*2*np.pi) ** 2))
                #add to parameters
                self.parameters.add('C', value=cap_ideal, min=cap_ideal * 0.8, max=cap_ideal* 1, vary=True)
                self.parameters.add('R_Fe', value=max_R_Fe, min=min_R_Fe, max=max_R_Fe, vary=True)
                #main element
                self.parameters.add('L', value=self.nominal_value, min=self.nominal_value * 0.9, max=self.nominal_value * 1.1, vary=False)
            case 2:
                # calculate "perfect" inductor for this resonance
                ind_ideal = 1 / (self.nominal_value * ((self.f0 * 2 * np.pi) ** 2))
                self.parameters.add('L', value=ind_ideal, min=ind_ideal * 0.8, max=ind_ideal * 1)
                self.parameters.add('R_iso', value=max_R_iso, min=min_R_iso, max=max_R_iso, vary=True)
                #main element
                self.parameters.add('C', value=self.nominal_value, min=self.nominal_value * 0.7,
                                    max=self.nominal_value * 1.1, vary=False)
            case 3:
                #TODO: CMCs -> eh scho wissen
                dummy = 0

        return 0

    def create_elements(self):

        #if we got too many frequency zones -> restrict fit to max order
        #else get order from frequency zones and write found order to class
        if self.max_order > len(self.frequency_zones):
            order = len(self.frequency_zones)
            self.order = len(self.frequency_zones)
        else:
            order = self.max_order
            #TODO: and also throw and except please

        C = self.parameters['C'].value
        L = self.parameters['L'].value

        min_cap = 1e-12
        max_cap = C * 1e3
        value_cap = (max_cap-min_cap)/2



        for key_number in range(1, order + 1):

            #create keys
            C_key   = "C%s" % key_number
            L_key   = "L%s" % key_number
            R_key   = "R%s" % key_number
            w_key   = "w%s" % key_number
            BW_key  = "BW%s" % key_number

            # get frequencies for the band
            f_l = self.frequency_zones[key_number-1][0] #lower
            f_c = self.frequency_zones[key_number-1][1] #center
            f_u = self.frequency_zones[key_number-1][2] #upper

            # bandwidth (formulas copied from payers script)
            log_multiplication_factor = 1.04
            BW_min      = f_c * log_multiplication_factor - f_c / log_multiplication_factor
            BW_max      = (f_u - f_l) * 1.1
            BW_value    = BW_max / 8

            #center frequency (omega)
            w_c = f_c * 2 * np.pi
            min_w = np.sqrt( (f_l*2*np.pi) * w_c)
            max_w = np.sqrt( (f_u*2*np.pi) * w_c)

            # expression strings
            expression_string_L = '1/(' + w_key + '**2*' + C_key + ')'

            match self.fit_type: #TODO:check if fit_type is valid!!!
                case 1: #INDUCTOR
                    expression_string_R = '1/(2*' + str(np.pi) + '*' + BW_key + '*' + C_key + ')'
                case 2:
                    expression_string_R = '2*' + str(np.pi) + '*' + BW_key + '*' + L_key

            #add parameters
            self.parameters.add(BW_key, min=BW_min,     max=BW_max,     value=BW_value)
            self.parameters.add(w_key,  min=min_w,      max=max_w,      value=w_c)
            self.parameters.add(C_key,  min=min_cap,    max=max_cap,    value=value_cap)
            self.parameters.add(L_key,  min=1e-20,      max=L,          expr=expression_string_L)
            self.parameters.add(R_key,  min=1e-3,       max=1e4,        expr=expression_string_R)
        dummy = 2
        return 0

    def calculate_Z(self, parameters, frequency_vector, data):
        #method to calculate the impedance curve from chained parallel resonance circuits
        #this method is needed for the fitter

        #if we only want to fit the main resonant circuit, set order to zero to avoid "for" loops
        fit_main_res = 1
        if fit_main_res:
            order = 0
        else:
            order = self.max_order

        #create array for frequency
        freq = frequency_vector
        w = freq * 2 * np.pi

        #get parameters for main circuit
        C = parameters['C'].value
        L = parameters['L'].value
        R_s = parameters['R_s'].value

        match self.fit_type:
            case 1:
                R_Fe = parameters['R_Fe'].value
            case 2:
                R_iso = parameters['R_iso'].value

        #calculate main circuits resistance
        XC = 1 / (1j * w * C)
        XL = 1j * w * L
        Z = 0
        match self.fit_type:
            case 1: #INDUCTOR
                Z_part1 = 1 / ((1 / R_Fe) + (1 / XL))
                Z_main = 1 / ((1 / (R_s + Z_part1)) + (1 / XC))
            case 2: #CAPACITOR
                Z_main = (1 / ((1 / R_iso) + (1 / XC))) + XL + R_s


        Z = Z_main


        diff = (np.real(data) - np.real(Z)) + 1j * (np.imag(data) - np.imag(Z))

        return abs(diff)

    def start_fit(self):
        fit_main_resonance = 1
        freq = self.files[0].data.f
        #this is copied from paier's code; find the index where the main resonance lies
        #this is required in order to fit the main resonance circuit, otherwise the fit for the main circuit will not
        #work since the resonances in the higher frequencies can't be modeled by the main circuit
        #TODO: maybe solve this via boolean indexing as well
        for post_resonance_range in range(len(self.data_ang)):
            if post_resonance_range + 10 >= len(freq):
                break
            if np.sign(self.data_ang[post_resonance_range]) != np.sign(self.data_ang[post_resonance_range + 10]):
                break
        #TODO: we are using only one data point for the fit? seems a bit weird, but if an array is used, it does not seem to work
        fit_data = self.z21_data[post_resonance_range]
        # out_base_model = minimize(self.calculate_Z, self.parameters, kws={'frequency_vector': freq[:post_resonance_range],
        #                                                                   'fit_main_res': fit_main_resonance,
        #                                                                   'data':fit_data},
        #                           method='powell', options={'xtol': 1e-18, 'disp': True})

        out_base_model = minimize(self.calculate_Z, self.parameters, args=(freq[:post_resonance_range], fit_data),
                                  method='powell', options={'xtol': 1e-18, 'disp': True})

        self.out = out_base_model
        return 0







