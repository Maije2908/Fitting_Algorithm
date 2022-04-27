
# The fitter class shall take the task of fitting the data, as well as smoothing it and performing manipulations
# This is likely to become a rather long task, especially for CMCs and this class is therefore likely to be long
# I do not know yet what it will have to contain and how to best handle the data
# Most of this class will be based on Payer's program
#NOTE: THE CLASS IN THE FORM THAT IT IS NOW IS NOT ABLE TO MANAGE MULTIPLE FILES!!!!!



import matplotlib.pyplot as plt
import numpy as np
import scipy
from lmfit import minimize, Parameters
from scipy.signal import find_peaks

# import constants into the same namespace
import fitterconstants
from fitterconstants import *


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
        self.model_data = None

        self.fit_type = None
        self.out = None
        #using the Parameters() class from lmfit, might be necessary to make this an array when handling multiple files
        self.parameters = Parameters()

        self.frequency_zones = None
        self.bandwidths = None
        self.peak_heights = None
        self.frequency_vector = None


        #TODO: maybe change this; the f0 variable is for testing puropses
        self.f0 = None
        self.max_order = fitterconstants.MAX_ORDER
        self.order = None


    ####################################################################################################################
    # Parsing Methods
    ####################################################################################################################

    #method to set the entry values of the specification
    def set_specification(self, pass_val, para_r, prom, sat, fit_type):

        #TODO: maybe do some error handling here? although it is safe to assume that we get good fit_type values
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
            self.prominence = fitterconstants.PROMINENCE_DEFAULT
        else:
            self.prominence = prom

        if sat is None:
            self.saturation = None
        else:
            self.saturation = sat



    #method to parse the files from the iohandler
    def set_files(self, files):
        self.files = files
        try:
            self.frequency_vector = self.files[0].data.f
        except Exception as e:
            pass



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
        offset = 0  # samples
        nominal_value = 0
        freq = self.frequency_vector

        match self.fit_type:
            case El.INDUCTOR:

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

            case El.CAPACITOR:

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
        freq = self.frequency_vector

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

        min_prominence_phase = fitterconstants.PROMINENCE_DEFAULT
        prominence_mag = 0.01
        R_s = self.parasitive_resistance
        freq = self.frequency_vector
        prominence_phase = max(min_prominence_phase, float(self.prominence))
        #TODO: maybe delete this(is for testing)
        prominence_mag = prominence_phase


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

        min_zone_start = self.f0 * 3  # frequency buffer for first RLC circuit TODO: AD "find_peaks" this might do the trick

        # TODO: delete unnecessary variables here
        ang_minima_pos = f_phase_minima[f_phase_minima > min_zone_start]
        ang_maxima_pos = f_phase_maxima[f_phase_maxima > min_zone_start]

        ang_minima_pos = f_mag_minima[f_mag_minima > min_zone_start]
        ang_maxima_pos = f_mag_maxima[f_mag_maxima > min_zone_start]

        mag_minima_pos = f_mag_minima[f_mag_minima > min_zone_start]
        mag_maxima_pos = f_mag_maxima[f_mag_maxima > min_zone_start]

        mag_minima_index = mag_minima[0][f_mag_minima > min_zone_start]
        mag_maxima_index = mag_maxima[0][f_mag_maxima > min_zone_start]

        mag_maxima_value = mag_maxima[1]['peak_heights'][f_mag_maxima > min_zone_start]


        # plot commands to check peak values TODO: this is for testing
        # markerson = mag_maxima[0]
        # plt.loglog(self.data_mag,'-bD', markevery=markerson)
        # plt.show()
        # plt.figure()

        number_zones = len(mag_minima_pos)
        bandwidth_list = []
        peak_heights = []
        for num_maximum in range(0, number_zones):
            #resonance frequency, corresponding height and index
            res_fq = ang_maxima_pos[num_maximum]
            res_index = mag_maxima_index[num_maximum]
            res_value = mag_maxima_value[num_maximum]

            #get 3dB value
            bw_value = res_value / np.sqrt(2)

            #find the next point on the curve where the 3dB value is reached
            #we need to flip the array for the lower value
            try:
                f_lower_index = np.flipud(np.argwhere(np.logical_and(freq < res_fq, self.data_mag < bw_value)))[0][0]
            except IndexError:
                f_lower_index = res_index - fitterconstants.DEFAULT_OFFSET_PEAK

            try:
                f_upper_index = np.argwhere(np.logical_and(freq > res_fq, self.data_mag < bw_value))[0][0]
            except IndexError:
                #here we need to account for the fact that we could overshoot the max index
                if res_index + fitterconstants.DEFAULT_OFFSET_PEAK < len(freq):
                    f_upper_index = res_index + fitterconstants.DEFAULT_OFFSET_PEAK
                else:
                    f_upper_index = len(freq) - 1

            # check if the found 3dB points are in an acceptable range i.e. not "behind" the next peak or "in front of"
            # the previous peak. If that is the case we set the index to a default offset to get a "bandwidth"
            if num_maximum != 0:
                if f_lower_index < mag_maxima_index[num_maximum - 1]:
                    f_lower_index = res_index - fitterconstants.DEFAULT_OFFSET_PEAK
            if num_maximum < number_zones-1:
                if f_upper_index > mag_maxima_index[num_maximum + 1]:
                    #again we could overshoot the max index here
                    if res_index + fitterconstants.DEFAULT_OFFSET_PEAK < len(freq):
                        f_upper_index = res_index + fitterconstants.DEFAULT_OFFSET_PEAK
                    else:
                        f_upper_index = len(freq) - 1

            if (self.data_mag[res_index] < self.data_mag[f_upper_index] ) or (self.data_mag[res_index] < self.data_mag[f_lower_index]):
                #TODO: delete peak in that case; EDIT: not actually necessary if we write to the list only if that condition is not fulfilled
                pass
            else:
                f_tuple = [freq[f_lower_index], res_fq, freq[f_upper_index]]
                bandwidth_list.append(f_tuple)
                peak_heights.append(abs(res_value))
                #THIS IS FOR TESTING
                markerson = [f_lower_index,res_index,f_upper_index]
                plt.loglog(self.data_mag, '-bD', markevery=markerson)

        #spread BW of last circuit; TODO: maybe center the band?
        strech_factor = 5
        end_zone_offset_factor = 3
        # bandwidth_list[-1][0] = max(freq) * (1/strech_factor)
        bandwidth_list[-1][1] = max(freq) * end_zone_offset_factor
        bandwidth_list[-1][2] = max(freq) * strech_factor
        bandwidth_list[-1][0] = mag_minima_pos[-1] #bandwidth_list[-1][1] - (bandwidth_list[-1][2] - bandwidth_list[-1][1])
        peak_heights[-1] = abs(max(self.z21_data)) * 2


        # f1 = f_mag_maxima[-2]
        # f3 = max(freq) * 3
        # f2 = max(freq) * 1.2
        # f_tuple = [f1,f2,f3]
        # bandwidth_list.append(f_tuple)
        # peak_heights.append(abs(max(self.z21_data)))
        #
        #
        # # #loop to find frequency ranges, copied from payer
        # number_zones = len(ang_minima_pos)
        # f_zones_list = []
        # for num_minimum in range(0, number_zones):
        #     f1 = ang_minima_pos[num_minimum]
        #     f3 = max(freq) * 5
        #     if num_minimum + 1 < number_zones:
        #         f3 = ang_minima_pos[num_minimum + 1]
        #
        #     # find the maxima between two minima
        #     for num_maximum in range(len(ang_maxima_pos)):
        #         if f1 < ang_maxima_pos[num_maximum] < f3:
        #             f_tuple = (f1, ang_maxima_pos[num_maximum], f3)
        #             f_zones_list.append(f_tuple)
        #             # #TODO: this is also for testing
        #             # temp_d = abs(self.data_mag[np.logical_and(freq > f1, freq < f3)])
        #             # temp_f = freq[np.logical_and(freq > f1, freq < f3)]
        #             # plt.loglog(temp_f,temp_d)
        #
        #             break  # corresponding f2 found
        # try:
        #     if ang_minima_pos[-1] > ang_maxima_pos[-1]:
        #         f_tuple = (ang_minima_pos[-1], max(freq) * 3, f3)
        #         f_zones_list.append(f_tuple)
        # # no minima or maxima present - not sure if this works correctly TODO: me neither, but let's assume it works for the moment
        # except Exception as e:
        #     if number_zones > 0:  # else base model
        #         f_tuple = (ang_minima_pos[-1], np.sqrt(max(freq) ** 2 * 3), max(freq) * 6)
        #         f_zones_list.append(f_tuple)
        #         print("Warning from frequency zones: {e}".format(e=e))
        #         pass

        # self.frequency_zones = f_zones_list
        self.peak_heights = peak_heights
        self.bandwidths = bandwidth_list

    def create_nominal_parameters(self):

        self.parameters.add('R_s', value=self.parasitive_resistance, min=1e-20, vary=False)




        #get bandwidth
        #TODO:also do this for capacitors!!!
        freq = self.frequency_vector
        res_value = self.z21_data[freq == self.f0]
        bw_value = res_value / np.sqrt(2)
        f_lower_index = np.flipud(np.argwhere(np.logical_and(freq < self.f0, self.data_mag < bw_value)))[0][0]
        f_upper_index = (np.argwhere(np.logical_and(freq > self.f0, self.data_mag < bw_value)))[0][0]
        BW = freq[f_upper_index]-freq[f_lower_index]

        R_Fe = (self.f0 * (self.f0*2*np.pi)*self.nominal_value) / BW

        match self.fit_type:
            case fitterconstants.El.INDUCTOR:
                #calculate "perfect" capacitor for this resonance
                cap_ideal = 1 / (self.nominal_value * ((self.f0*2*np.pi) ** 2))
                #add to parameters
                self.parameters.add('C', value=cap_ideal, min=cap_ideal * fitterconstants.MAIN_RES_PARASITIC_LOWER_BOUND,
                                    max=cap_ideal * fitterconstants.MAIN_RES_PARASITIC_UPPER_BOUND, vary=True)

                self.parameters.add('R_Fe', value=R_Fe, min=fitterconstants.MIN_R_FE, max=fitterconstants.MAX_R_FE, vary=True)
                #main element
                self.parameters.add('L', value=self.nominal_value, min=self.nominal_value * 0.9, max=self.nominal_value * 1.1, vary=False)
            case fitterconstants.El.CAPACITOR:
                # calculate "perfect" inductor for this resonance
                ind_ideal = 1 / (self.nominal_value * ((self.f0 * 2 * np.pi) ** 2))
                self.parameters.add('L', value=ind_ideal, min=ind_ideal * fitterconstants.MAIN_RES_PARASITIC_LOWER_BOUND,
                                    max=ind_ideal * fitterconstants.MAIN_RES_PARASITIC_UPPER_BOUND)
                self.parameters.add('R_iso', value=fitterconstants.MAX_R_ISO, min=fitterconstants.MIN_R_ISO, max=fitterconstants.MAX_R_ISO, vary=True)
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
        if self.max_order > len(self.bandwidths):
            order = len(self.bandwidths)
            self.order = len(self.bandwidths)
        else:
            order = self.max_order
            #TODO: and also throw and except please
            #TODO: some methods are not robust enough for this fit maybe?

        C = self.parameters['C'].value
        L = self.parameters['L'].value

        min_cap = fitterconstants.MIN_CAP
        max_cap = C * fitterconstants.MAX_CAP_FACTOR
        value_cap = (max_cap-min_cap)/2



        for key_number in range(1, order + 1):

            #create keys
            C_key   = "C%s" % key_number
            L_key   = "L%s" % key_number
            R_key   = "R%s" % key_number
            w_key   = "w%s" % key_number
            BW_key  = "BW%s" % key_number

            # get frequencies for the band
            # f_l = self.frequency_zones[key_number-1][0] #lower
            # f_c = self.frequency_zones[key_number-1][1] #center
            # f_u = self.frequency_zones[key_number-1][2] #upper

            b_l = self.bandwidths[key_number-1][0]
            b_c = self.bandwidths[key_number-1][1]
            b_u = self.bandwidths[key_number-1][2]

            # bandwidth
            BW_min      = (b_u - b_l) * fitterconstants.BW_MIN_FACTOR
            BW_max      = (b_u - b_l) * fitterconstants.BW_MAX_FACTOR
            BW_value    = (b_u - b_l)  # BW_max / 8


            # center frequency (omega)
            w_c = b_c * 2 * np.pi
            min_w = b_l*2*np.pi * fitterconstants.MIN_W_FACTOR #np.sqrt( (f_l*2*np.pi) * w_c)
            max_w = b_u*2*np.pi * fitterconstants.MAX_W_FACTOR #np.sqrt( (f_u*2*np.pi) * w_c)

            # calculate Q-factor
            q = b_c/BW_value
            # take the value of the peak as the parallel resistor
            r_value = self.peak_heights[key_number-1]


            # if key_number <= order:
            #     r_value = abs(self.z21_data[np.argwhere(self.frequency_vector == f_c)])[0][0]
            # else:
            #     #the last frequency zone is not really a resonant circuit, so we have to account for that
            #     r_value = abs(self.z21_data[-1])

            # shrink beginning of first zone (each step halves range on log scale)
            # if key_number == 1:
            #     max_w = np.sqrt(max_w * w_c)
            #     for _ in range(4):
            #         min_w = np.sqrt(min_w * w_c)

            # expression strings
            expression_string_L = '1/(' + w_key + '**2*' + C_key + ')'
            expression_string_C = '1/(' + w_key + '**2*' + L_key + ')'

            L_value = 1 / (w_c**2 * value_cap)

            match self.fit_type: #TODO:check if fit_type is valid!!!
                case fitterconstants.El.INDUCTOR: #INDUCTOR
                    expression_string_R = '1/(2*' + str(np.pi) + '*' + BW_key + '*' + C_key + ')'
                case fitterconstants.El.CAPACITOR:
                    expression_string_R = '2*' + str(np.pi) + '*' + BW_key + '*' + L_key

            #testing new expression strings for L and C dependent on Q-factor
            expression_string_C = '(' + w_key + '/ (2*' + str(np.pi) + '))' + '/' + '(' + BW_key + '*' + w_key+ '*' + R_key + ')'
            expression_string_L = '(' + BW_key + '*'+ R_key + ')' +'/ ((' + w_key + '/ (2*' + str(np.pi) + ')) *' + w_key + ')'

            value_cap = (w_c / (2 * np.pi)) / (BW_value * w_c * r_value)
            value_ind = (BW_value * r_value) / ( (w_c / (2 * np.pi)) * w_c)

            max_cap = value_cap * 2
            min_cap = value_cap * 0.5

            min_ind = value_ind * 2
            max_ind = value_ind *0.5

            #add parameters

            #this is the default configuration, i.e. the config how tristan had it
            # self.parameters.add(BW_key, min=BW_min,     max=BW_max,     value=BW_value              , vary=True)
            # self.parameters.add(w_key,  min=min_w,      max=max_w,      value=w_c                   , vary=True)
            # self.parameters.add(C_key,  min=min_cap,    max=max_cap,    value=value_cap             , vary=True)
            # self.parameters.add(L_key,  min=1e-20,      max=L,          expr=expression_string_L    , vary=False)
            # self.parameters.add(R_key,  min=1e-3,       max=1e4,        expr=expression_string_R    , vary=False)

            #just vary the last resonant frequency since all other resonances are well determined
            #the last "resonance" is not really a resonance but rather the "end of data"
            if key_number == order:
                expression_string_L = '1/(' + w_key + '**2*' + C_key + ')'
                #
                #vary parameters so that none of them get too small
                if (1/(w_c**2 * value_cap)) < fitterconstants.MINIMUM_PRECISION:
                    while (1/(w_c**2 * value_cap)) < fitterconstants.MINIMUM_PRECISION:
                        value_cap = value_cap/10
                    value_ind = 1 / (w_c ** 2 * value_cap)
                elif (1/(w_c**2 * value_ind)) < fitterconstants.MINIMUM_PRECISION:
                    while (1/(w_c**2 * value_ind)) < fitterconstants.MINIMUM_PRECISION:
                        value_ind = value_ind/10
                    value_cap = 1 / (w_c ** 2 * value_ind)

                max_cap = value_cap * 2
                min_cap = value_cap * 0.5
                min_ind = value_ind * 2
                max_ind = value_ind * 0.5
                expression_string_p = '1/(2*' + str(np.pi) + '*' + BW_key + '*' + C_key + ')'
                BW_min = b_c*1.04 -b_c/1.04
                BW_max = (b_u - b_l) * 1.1


                self.parameters.add(BW_key, min=BW_min, max=BW_max, value=BW_max/8, vary=True)
                self.parameters.add(w_key, min=min_w, max=max_w, value=w_c, vary=True)

                self.parameters.add(C_key, min=min_cap, max=max_cap, value=value_cap, vary=True)
                self.parameters.add(L_key, min=min_ind, max=max_ind, expr=expression_string_L , vary=True)
                self.parameters.add(R_key, min=fitterconstants.RMIN, max=fitterconstants.RMAX, expr=expression_string_p, vary=True)
            else:

                self.parameters.add(w_key, min=min_w, max=max_w, value=w_c, vary=False)
                self.parameters.add(R_key, min=fitterconstants.RMIN, max=fitterconstants.RMAX, value=r_value,vary=True)
                self.parameters.add(BW_key, min=BW_min, max=BW_max, value=BW_value, vary=True)
                self.parameters.add(C_key, min=min_cap, max=max_cap, expr=expression_string_C, vary=False)
                self.parameters.add(L_key, min=fitterconstants.LMIN, max=L, expr=expression_string_L, vary=False)
                # self.parameters.add(C_key, min=min_cap, max=max_cap, value=value_cap, vary=True)
                # self.parameters.add(L_key, min=min_ind, max=max_ind, value=value_ind, vary=True)

            # self.parameters.add(BW_key, min=BW_min, max=BW_max, value=BW_value, vary=True)
            # self.parameters.add(C_key, min=min_cap, max=max_cap, value=value_cap, vary=True)
            # self.parameters.add(L_key, min=fitterconstants.LMIN, max=L, expr=expression_string_L, vary=False)
            # # self.parameters.add(L_key, min=fitterconstants.LMIN, max=L, value=L_value, vary=True)
            # self.parameters.add(R_key, min=fitterconstants.RMIN, max=fitterconstants.RMAX, expr=expression_string_R, vary=False)


        return 0




    def calculate_Z(self, parameters, frequency_vector, data, fit_order, fit_main_res, modeflag):
        #method to calculate the impedance curve from chained parallel resonance circuits
        #this method is needed for the fitter

        #if we only want to fit the main resonant circuit, set order to zero to avoid "for" loops
        if fit_main_res:
            order = 0
        else:
            order = fit_order #TODO: this method could be called before the order is set

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
            case El.INDUCTOR: #INDUCTOR
                Z_part1 = 1 / ((1 / R_Fe) + (1 / XL))
                Z_main = 1 / ((1 / (R_s + Z_part1)) + (1 / XC))
            case El.CAPACITOR: #CAPACITOR
                Z_main = (1 / ((1 / R_iso) + (1 / XC))) + XL + R_s

        Z = Z_main

        for actual in range(1, order + 1):
            key_number = actual
            C_key = "C%s" % key_number
            L_key = "L%s" % key_number
            R_key = "R%s" % key_number
            C_act = parameters[C_key].value
            L_act = parameters[L_key].value
            R_act = parameters[R_key].value
            Z_C   = 1 / (1j * w * C_act)
            Z_L   = (1j * w * L_act)
            Z_R   = R_act

            Z    += 1 / ( (1/Z_C) + (1/Z_L) + (1/Z_R) )

        # diff = (np.real(data) - np.real(Z)) + 1j * (np.imag(data) - np.imag(Z))
        # return abs(diff)

        match modeflag:
            case fcnmode.FIT:
                diff = (np.real(data) - np.real(Z)) + (np.imag(data) - np.imag(Z))
                #diff = (np.real(data) - np.real(Z))**2 + (np.imag(data) - np.imag(Z))**2
                return abs(diff)
            case fcnmode.OUTPUT:
                return Z


    def start_fit(self):

        freq = self.frequency_vector
        fit_data = self.z21_data
        self.create_elements()
        fit_order = self.order
        fit_main_resonance = 0

        # mode = fitterconstants.fcnmode.OUTPUT
        #
        # model_data_before_fit = self.calculate_Z(self.parameters, freq, [2], self.order, fit_main_resonance, mode)

        # self.fit_end_zone()

        mode = fitterconstants.fcnmode.FIT

        # fit_main_resonance = 1
        #
        # self.out = minimize(self.calculate_Z, self.parameters,
        #                args=(freq, fit_data, fit_order, fit_main_resonance, mode,),
        #                method='powell', options={'xtol': 1e-18, 'disp': True})
        #
        # test_model_data = self.calculate_Z(self.out.params, freq, [2], self.order, fit_main_resonance, mode)
        # plt.loglog(self.frequency_vector, abs(fit_data))
        # plt.loglog(self.frequency_vector, abs(test_model_data))
        #
        # self.overwrite_main_resonance_parameters()



        self.out = minimize(self.calculate_Z, self.parameters,
                       args=(freq, fit_data, fit_order, fit_main_resonance, mode,),
                       method='powell', options={'xtol': 1e-18, 'disp': True})

        #TODO: here we have the model -> do some output here


        #calculate output
        self.out.params.pretty_print()
        mode = fitterconstants.fcnmode.OUTPUT
        #freq = np.linspace(min(freq),max(freq)+1e9,12000)
        model_data = self.calculate_Z(self.out.params,freq,[2],self.order,fit_main_resonance,mode)

        self.model_data = model_data



        #for testin purposes
        plt.figure()
        plt.loglog(self.frequency_vector, abs(fit_data))
        plt.loglog(freq, abs(model_data))
        plt.show()

        self.do_output()



        return 0

    def do_output(self):

        title = "test"
        fig = plt.figure(figsize=(20, 20))
        #file_title = get_file_path.results + '/03_Parameter-Fitting_' + file_name + "_" + mode
        plt.subplot(311)
        plt.title(str(title), pad=20, fontsize=25, fontweight="bold")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([min(self.frequency_vector), max(self.frequency_vector)])
        plt.ylabel('Magnitude in \u03A9', fontsize=16)
        plt.xlabel('Frequency in Hz', fontsize=16)
        plt.grid(True, which="both")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(self.frequency_vector, abs(self.z21_data), 'r', linewidth=3, alpha=0.33, label='Measured Data')
        plt.plot(self.frequency_vector, self.data_mag, 'r', linewidth=3, alpha=1, label='Filtered Data')
        # Plot magnitude of model in blue
        plt.plot(self.frequency_vector, abs(self.model_data), 'b--', linewidth=3, label='Model')
        plt.legend(fontsize=16)

        #Phase
        curve = np.angle(self.z21_data, deg=True)

        plt.subplot(312)
        plt.xscale('log')
        plt.xlim([min(self.frequency_vector), max(self.frequency_vector)])
        plt.ylabel('Phase in °', fontsize=16)
        plt.xlabel('Frequency in Hz', fontsize=16)
        plt.grid(True, which="both")
        plt.yticks(np.arange(45 * (round(min(curve) / 45)), 45 * (round(max(curve) / 45)) + 1, 45.0))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(self.frequency_vector, np.angle(self.z21_data, deg=True), 'r', linewidth=3, zorder=-2, alpha=0.33, label='Measured Data')
        plt.plot(self.frequency_vector, self.data_ang, 'r', linewidth=3, zorder=-2, alpha=1, label='Filtered Data')
        #   Plot Phase of model in magenta
        plt.plot(self.frequency_vector, np.angle(self.model_data, deg=True), 'b--', linewidth=3, label='Model', zorder=-1)
        #plt.scatter(resonances_pos, np.zeros_like(resonances_pos) - 90, linewidth=3, color='green', s=200, marker="2",
        #            label='Resonances')
        plt.legend(fontsize=16)
        # plt.savefig(file_title)
        # plt.close(fig)







    #################################### V OBSOLETE V###################################################################


    def fit_end_zone(self):
        #method to fit the slope at the end of data and fix it... this is just a test, so it might not work as intended
        temp_params = Parameters()
        key_number = self.order
        C_key = "C%s" % key_number
        L_key = "L%s" % key_number
        R_key = "R%s" % key_number
        temp_params.add("C")
        temp_params.add("L")
        temp_params.add("R_Fe")
        temp_params.add("R_s", value = 0, vary = False)

        offset = 2700

        temp_params["C"] = self.parameters[C_key]
        temp_params["L"] = self.parameters[L_key]
        temp_params["R_Fe"] = self.parameters[R_key]

        freq = self.frequency_vector[offset:-1]
        fit_data = self.z21_data[offset:-1]
        fit_order = 0
        fit_main_resonance = 1
        mode = fitterconstants.fcnmode.FIT

        end_zone_model = minimize(self.calculate_Z, temp_params,
                       args=(freq, fit_data, fit_order, fit_main_resonance, mode,),
                       method='leastsq')

        mode = fitterconstants.fcnmode.OUTPUT
        freq = np.linspace(min(self.frequency_vector),max(self.frequency_vector)+1e12,120000)
        model_data = self.calculate_Z(end_zone_model.params,freq, [2], self.order, fit_main_resonance, mode)

        plt.figure()
        plt.loglog(freq, abs(model_data))
        plt.loglog(self.frequency_vector, abs(self.z21_data))
        plt.show()


    def overwrite_main_resonance_parameters(self):
        #method to overwrite the nominal parameters with the parameters obtained by modeling the main resonance circuit
        #the method is essentially the same as the "create nominal parameters"

        self.parameters.add('R_s', value=self.out.params['R_s'].value, min=1e-20, vary= False)

        #unfortuinately we still need the limits for the parameters
        # max/min values for the isolation/iron resistance
        min_R_Fe = 10
        max_R_Fe = 1e9
        min_R_iso = 1
        max_R_iso = 1e12

        #note that it might not be necessary to re-set parameters that have the "vary" boolean set to "false"
        #but it might be better readable like this
        #TODO: The "vary" bools are all set to false after we have fit the main resonance, so there might not be the need
        #TODO: to do this part in such a verbose style -> i.e. make the code slimmer here
        match self.fit_type:
            case 1:
                cap_ideal = 1 / (self.nominal_value * ((self.f0 * 2 * np.pi) ** 2))
                self.parameters.add('C', value=self.out.params['C'].value, min=cap_ideal * 0.8, max=cap_ideal* 1, vary=False)
                self.parameters.add('R_Fe', value=self.out.params['R_Fe'].value, min=min_R_Fe, max=max_R_Fe, vary=False)
                # main element
                self.parameters.add('L', value=self.out.params['L'].value, min=self.nominal_value * 0.9, max=self.nominal_value * 1.1, vary=False)
            case 2:
                ind_ideal = 1 / (self.nominal_value * ((self.f0 * 2 * np.pi) ** 2))
                self.parameters.add('L', value=self.out.params['L'].value, min=ind_ideal * 0.8, max=ind_ideal * 1)
                self.parameters.add('R_iso', value=self.out.params['R_iso'].value, min=min_R_iso, max=max_R_iso, vary=False)
                # main element
                self.parameters.add('C', self.out.params['C'].value, min=self.nominal_value * 0.7, max=self.nominal_value * 1.1, vary=False)
            case 3:
                # TODO: CMCs -> eh scho wissen
                dummy = 0

        return 0


    def test_fit_main_res(self):
        #function for testing purposes NOT IN USE

        C = self.out.params['C'].value
        L = self.out.params['L'].value
        R_Fe = self.out.params['R_Fe'].value
        R_s = self.out.params['R_s'].value

        w = self.frequency_vector * 2 * np.pi
        XC = 1 / (1j * w * C)
        XL = 1j * w * L
        Z = 0
        Z_part1 = 1 / ((1 / R_Fe) + (1 / XL))
        Z_main = 1 / ((1 / (R_s + Z_part1)) + (1 / XC))
        plt.loglog(Z_main)
        plt.loglog(self.data_mag)


    def fit_iteration_callback(self, out_model):
        #method to set the parameters of the output to the parameters used for the fit TODO:rewrite this comment
        #TODO: delete -> UNUSED METHOD
        for key in self.parameters.keys():
            self.parameters[key].value = out_model.params[key].value
            self.parameters[key].min = self.parameters[key] * 0.8
            self.parameters[key].max = self.parameters[key] * 1.2






