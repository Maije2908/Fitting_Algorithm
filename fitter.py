




import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal as sg
from lmfit import minimize, Parameters
from scipy.signal import find_peaks
import decimal
import copy
import sys
import pandas as pd


# import constants
import constants
from constants import *
import config
# constants for this class
FIT_BY = config.FIT_BY


class Fitter:

    def __init__(self, logger_instance):
        self.nominal_value = None
        self.series_resistance = None
        self.prominence = constants.PROMINENCE_DEFAULT
        self.saturation = None

        self.file = None

        self.z21_data = None
        self.data_mag = None
        self.data_ang = None
        self.model_data = None

        self.fit_type = None
        self.ser_shunt = None
        self.captype = None

        self.out = None

        self.parameters = Parameters()

        self.logger = logger_instance

        self.frequency_zones = None
        self.bandwidths = None
        self.modeled_bandwidths = []
        self.bad_bandwidth_flag = None
        self.peak_heights = None
        self.frequency_vector = None


        self.acoustic_resonant_frequency = None
        self.f0 = None
        self.f0_index = None

        self.order = 0

        self.offset = 0



    ####################################################################################################################
    # Parsing Methods
    ####################################################################################################################

    def set_specification(self, pass_val, para_r, prom, fit_type, captype = None):

        """

        Function to pass specification of the DUT to the fitter.

        Specification can be given in the GUI but default values are chosen if some of the parameters are not given;
        if the nominal value of the DUT is not given, it will be calculated.

        :param pass_val:    value of inductance/capacitance of the DUT in Henry/Farad
        :param para_r:      parasitive resistance of the DUT in Ohms
        :param prom:        prominence for peak detection in dB
        :param fit_type:    type of the DUT; can be Inductor or Capacitor
        :param captype:     type of the Capacitor (if fit_type == Capacitor); can be MLCC or generic
        :return:            stores the given values in corresponding instance variables

        """

        self.fit_type = fit_type

        if fit_type == constants.El.CAPACITOR and not captype is None:
            self.captype = captype
        elif fit_type == constants.El.CAPACITOR:
            self.captype = constants.captype.GENERIC



        if para_r is None:
            self.calculate_nominal_Rs()
        else:
            self.series_resistance = para_r


        if pass_val is None:
            #if we do not have the nominal value try to calculate it
            try:
                self.calculate_nominal_value()
            #if we can't calculate it, pass the exception back to the calling function
            except Exception as e:
                raise e
        else:
            self.nominal_value = pass_val


        if prom is None:
            self.prominence = constants.PROMINENCE_DEFAULT
        else:
            self.prominence = prom


    def set_captype(self, captype):
        """
        Auxilliary method to set the Capacitor type.
        This is used in cases where a MLCC can't be calculated and the captype has to be reset to "Generic"

        :param captype:     Type of capacitor
        :return:            None (stores captype in corresponding instance variable)
        """
        self.captype = captype

    def set_file(self, file):
        """
        Method to set the file to the fitter

        :param file:        The File to be set (.s2p file in form of skrf.SNpFile())
        :return:            None (stores File in instance variable; also obtains frequency vector from the file)
        :raises Exception:  will raise an Exception ("No file provided") if frequency vector can't be obtained
        """

        self.file = file
        try:
            self.frequency_vector = self.file.data.f
            self.logger.info("File: " + self.file.name)
        except Exception:
            raise Exception("No Files were provided, please select a file!")




    ####################################################################################################################
    # Pre-Processing Methods
    ####################################################################################################################

    def calc_series_thru(self, Z0):
        """
        Function to calculate the series-through impedance of the DUT

        :param Z0: nominal impedance of the measurement system
        :return: None (stores calculated impedance in instance variable)
        """

        self.z21_data = 2 * Z0 * ((1 - self.file.data.s[:, 1, 0]) / self.file.data.s[:, 1, 0])
        self.ser_shunt = constants.calc_method.SERIES

    def calc_shunt_thru(self, Z0):
        """
        Function to calculate the shunt-through impedance of the DUT

        :param Z0: nominal impedance of the measurement system
        :return: None (stores impedance data in instance variable)
        """

        self.z21_data = (Z0 * self.file.data.s[:, 1, 0]) / (2 * (1 - self.file.data.s[:, 1, 0]))
        self.ser_shunt = constants.calc_method.SHUNT

    def smooth_data(self):
        """
        Function to smooth the impedance data

        A Savitzky-Golay filter is used to smooth the impedance data. The smoothed data will be stored in instance
        variables **data_mag** and **data_ang** for magnitude and phase respectively

        :return: None (stores smoothed data in instance variables data_mag and data_ang)
        """

        sav_gol_mode = 'interp'
        self.data_mag = scipy.signal.savgol_filter(abs(self.z21_data), constants.SAVGOL_WIN_LENGTH,
                                                   constants.SAVGOL_POL_ORDER, mode=sav_gol_mode)
        self.data_ang = scipy.signal.savgol_filter(np.angle(self.z21_data, deg=True), constants.SAVGOL_WIN_LENGTH,
                                                   constants.SAVGOL_POL_ORDER, mode=sav_gol_mode)
        #limit the phase data to +/- 90°
        self.data_ang = np.clip(self.data_ang, -90, 90)

    def calculate_nominal_value(self):
        """
        Function to calculate the nominal value of the DUT, if it was not provided.

        This works by looking for the linear range of the passive element, i.e. the range where the DUT behaves like a
        linear coil/cap. The value is then calculated by using the mean of all obtained values of the linear range

        :return:            Nominal value of the DUT in Henry/Farad (the value is also written to an instance variable)
        :raises Exception:  If the phase of the dataset is too low (i.e. inductive/capacitive linear range can not be
            detected); there is a constant that can be modified in the GUI_config to allow lower phase, however this might make
            the calculation less precise
        """

        offset = 0
        nominal_value = 0
        freq = self.frequency_vector

        match self.fit_type:
            case El.INDUCTOR:

                # calculate the offset; the data can be noisy at lower frequencies, leading to a lot of zero crossings
                # in the phase, which lead to misdetection of the main resonant frequency; hence we use a constant for
                # the minimum phase, everything lower than that will not be used for detection
                if sum(self.data_ang > constants.PHASE_OFFSET_THRESHOLD) > 0:
                    offset = np.argwhere(self.data_ang > constants.PHASE_OFFSET_THRESHOLD)[0][0]
                else:
                    offset = 0

                # find first point where the phase crosses 0 using numpy.argwhere --> f0
                index_angle_smaller_zero = np.argwhere(self.data_ang[offset:] < 0)

                # get the index of the zero crossing and the detected resonant frequency; offset has to be added here
                index_ang_zero_crossing = index_angle_smaller_zero[0][0] + offset
                f0 = freq[index_ang_zero_crossing]

                # check if the phase of the dataset has a valid range; if the phase is not around 90° we cannot assume
                # inductive range
                if max(self.data_ang[offset:index_ang_zero_crossing]) < constants.PERMITTED_MIN_PHASE:
                    #if we can't detect the nominal value raise exception
                    raise Exception("Error: Inductive range not detected (max phase = {value}°).\n"
                                    "Please specify nominal inductance.".format(value=np.round(max(self.data_ang), 1)))

                # crop data to [offset:f0] in order to obtain the linear range for the calculation of nominal value
                curve_data = self.z21_data[freq < f0][offset:]
                w_data = (freq[freq < f0][offset:])*2*np.pi

                # create an array filled with possible values for L; calculation is L = imag(Z)/omega
                L_vals = []
                for it, curve_sample in enumerate(zip(curve_data, w_data)):
                    L_vals.append(np.imag(curve_sample[0])/curve_sample[1])



                # calculate the slope of the magnitude and get the 50% quantile of it; after that find the max slope
                slope_quantile_50 = np.quantile(np.gradient(self.data_mag)[freq<f0],0.5)
                max_slope = slope_quantile_50 * constants.QUANTILE_MULTIPLICATION_FACTOR
                
                #boolean index the data that has lower than max slope and calculate the mean of it
                L_vals_mean = np.array(L_vals)[np.gradient(self.data_mag)[freq<f0][offset:] < max_slope]
                
                # finally write the obtained nominal value to the instance variable; also write back the offset
                self.nominal_value = np.mean(L_vals_mean)
                self.offset = offset
                output_dec = decimal.Decimal("{value:.3E}".format(value=self.nominal_value)) #TODO: this has to be normalized output to 1e-3/-6/-9 etc
                self.logger.info("Nominal Inductance not provided, calculated: " + output_dec.to_eng_string())


            case El.CAPACITOR:
                # calculation of the capacitance works analogue to the inductance
                if sum((self.data_ang < -constants.PHASE_OFFSET_THRESHOLD)) > 0:
                    offset = np.argwhere(self.data_ang < -constants.PHASE_OFFSET_THRESHOLD_CAPS)[0][0]
                else:
                    offset = 0

                # find first point where the phase crosses 0
                try:
                    index_angle_larger_zero = np.argwhere(self.data_ang[offset:] > 0)
                    index_ang_zero_crossing = index_angle_larger_zero[0][0] + offset
                    f0 = freq[index_ang_zero_crossing]
                except IndexError:
                    raise Exception("Error: could not determine main resonant frequency (no phase crossings found).\n"
                                    "You can try to adjust the PHASE_OFFSET_THRESHOLD_CAPS parameter to a smaller value")

                if min(self.data_ang[offset:index_ang_zero_crossing]) > -constants.PERMITTED_MIN_PHASE:
                    raise Exception("Error: Capacitive range not detected (min phase = {value}°).\n"
                                    "Please specify nominal capacitance.".format(value=np.round(min(self.data_ang), 1)))


                #crop data to [offset:f0] in order to find the linear range for the calculation of nominal value
                curve_data = self.z21_data[freq < f0][offset:]
                w_data = (freq[freq < f0][offset:])*2*np.pi

                #create an array filled with possible values for L; calculation is L = imag(Z)/w
                C_vals = []
                for it, curve_sample in enumerate(zip(curve_data, w_data)):
                    # if bool_select[it]:
                    C_vals.append(-1/(np.imag(curve_sample[0])*curve_sample[1]))


                # find the 50% quantile of the slope data and define the max slope allowed
                slope_quantile_50 = np.quantile(np.gradient(C_vals), 0.5)
                max_slope = slope_quantile_50 * constants.QUANTILE_MULTIPLICATION_FACTOR
                # boolean index the data that has lower than max slope and calculate the mean
                C_vals_eff = np.array(C_vals)[abs(np.gradient(C_vals)) < abs(max_slope)]
                self.nominal_value = np.mean(C_vals_eff)
                self.offset = offset
                output_dec = decimal.Decimal("{value:.3E}".format(value=self.nominal_value))
                self.logger.info("Nominal Capacitance not provided, calculated: " + output_dec.to_eng_string())

        return self.nominal_value

    def calculate_nominal_Rs(self):
        """
        Function to calculate the series resistance of the DUT if it was not provided.

        Here the minimum of the impedance data is taken; this is **very** imprecise for inductors but works very well
        for capacitors
        :return: obtained series resistance (also writes the value to an instance variable)
        """

        R_s_input = min(abs(self.data_mag))
        self.series_resistance = R_s_input
        output_dec = decimal.Decimal("{value:.3E}".format(value=R_s_input))
        self.logger.info("Nominal Resistance not provided, calculated: " + output_dec.to_eng_string())

        return self.series_resistance

    def get_main_resonance(self):
        """
        Method to calculate the main resonant frequency of the DUT.

        This works by looking for the zero crossing of the phase.

        :return: resonant frequency f0 in Hz (also writes the value and the index of the zero crossing to instance variables)
        :raises Exception: if the resonant frequency could not be determined
        """

        freq = self.frequency_vector

        #set w0 to 0 in order to have feedback, if the method didn't work
        w0 = 0

        match self.fit_type:

            case constants.El.INDUCTOR: #INDUCTOR
                testdata = self.data_ang[:]
                if sum(self.data_ang > constants.PHASE_OFFSET_THRESHOLD) > 0:
                    offset = np.argwhere(self.data_ang > constants.PHASE_OFFSET_THRESHOLD)[0][0]
                else:
                    offset = 0
                index_angle_smaller_zero = np.argwhere(self.data_ang[offset:] < 0)
                index_ang_zero_crossing = offset + index_angle_smaller_zero[0][0]
                continuity_check = index_angle_smaller_zero[10][0]

            case constants.El.CAPACITOR: #CAPACITOR
                if sum(self.data_ang < -constants.PHASE_OFFSET_THRESHOLD) > 0:
                    offset = np.argwhere(self.data_ang < -constants.PHASE_OFFSET_THRESHOLD)[0][0]
                else:
                    offset = 0

                try:
                    index_angle_larger_zero = np.argwhere(self.data_ang[offset:] > 0)
                    index_ang_zero_crossing = index_angle_larger_zero[0][0] + offset
                    continuity_check = index_angle_larger_zero[10][0]
                except IndexError:
                    raise Exception("Error: could not determine main resonant frequency (no phase crossings found).\n"
                                    "You can try to adjust the PHASE_OFFSET_THRESHOLD parameter to a smaller value")


        #write the calculated offset to the instance variable
        self.offset = offset


        if continuity_check:
            f0 = freq[index_ang_zero_crossing]
            w0 = f0 * 2 * np.pi
            self.f0 = f0
            self.f0_index = index_ang_zero_crossing
            #log and print
            output_dec = decimal.Decimal("{value:.3E}".format(value=f0))
            self.logger.info("Detected f0: "+ output_dec.to_eng_string())
            print("Detected f0: "+output_dec.to_eng_string())

        if constants.DEBUG_MULTIPLE_FITE_FIT:
            print(index_ang_zero_crossing)

        if w0 == 0:
            raise Exception('ERROR: Main resonant frequency could not be determined.')

        return f0

    def get_resonances(self):
        """
        Method to get the higher order resonances.

        Using scipy.find_peaks on the magnitude data, resonant peaks are obtained. Also the bandwidth will be detected
        and a flag will be set if the bandwidth could not be found (due to missing -3dB points)

        :return:    bandwidth list-list of the detected resonances containing lower, center and upper frequencies
            (upper and lower frequency are at the -3dB points); also stores the obtained bands, the peak heights, the order
            (i.e. total number of peaks) and a flag if the -3dB points are not present in instance variables
        :raises Exception: if no resonances have been found
        """


        # create one figure for debug plots
        if constants.DEBUG_BW_DETECTION:
            plt.figure()
            plt.title(self.file.name)

        freq = self.frequency_vector
        magnitude_data = self.data_mag
        phase_data = self.data_ang

        # frequency limit the data
        magnitude_data = magnitude_data[freq < config.FREQ_UPPER_LIMIT]
        phase_data = phase_data[freq < config.FREQ_UPPER_LIMIT]
        freq = freq[freq < config.FREQ_UPPER_LIMIT]

        # get prominence
        prominence_mag = self.prominence
        prominence_phase = self.prominence

        # find peaks of Magnitude Impedance curve (using scipy.signal.find_peaks)
        # using dB here in order to match the prominence values
        match self.ser_shunt:
            case constants.calc_method.SERIES:
                mag_maxima = find_peaks(20*np.log10(magnitude_data), prominence=prominence_mag)
            case constants.calc_method.SHUNT:
                mag_maxima = find_peaks(-20 * np.log10(magnitude_data), prominence=prominence_mag)


        # get frequency of acquired magnitude maxima
        f_mag_maxima = freq[mag_maxima[0]]

        #ignore all peaks that lie "before" the main resonance and that are to close to the main resonance
        min_zone_start = self.f0 * constants.MIN_ZONE_OFFSET_FACTOR

        mag_maxima_pos = f_mag_maxima[f_mag_maxima > min_zone_start]
        mag_maxima_index = mag_maxima[0][f_mag_maxima > min_zone_start]


        number_zones = len(mag_maxima_pos)
        bandwidth_list = []
        peak_heights = []
        bad_BW_flag = np.zeros((number_zones,2))

        for num_maximum in range(0, number_zones):
            # get index, frequency and value of impedance at that point
            res_fq = mag_maxima_pos[num_maximum]
            res_index = mag_maxima_index[num_maximum]
            res_value = magnitude_data[res_index]

            #get 3dB value
            match self.fit_type:
                case constants.El.INDUCTOR:
                    bw_value = res_value / np.sqrt(2)
                case constants.El.CAPACITOR:
                    bw_value = res_value*np.sqrt(2)

            try:
                #find the index where the 3db value is reached; also check if the frequency is lower than the resonance,
                #but higher than the min zone; if that does not work use the default offset
                #NOTE: since we need the first value in front of the resonance we have to flipud the array
                match self.fit_type:
                    case constants.El.INDUCTOR:
                        f_lower_index = np.flipud(np.argwhere(np.logical_and(freq > min_zone_start, np.logical_and(freq < res_fq, (magnitude_data) < (bw_value)))))[0][0]
                    case constants.El.CAPACITOR:
                        f_lower_index = np.flipud(np.argwhere(np.logical_and(freq > min_zone_start, np.logical_and(freq < res_fq, (magnitude_data) > (bw_value)))))[0][0]
            except IndexError:
                f_lower_index = res_index - constants.DEFAULT_OFFSET_PEAK
                bad_BW_flag[num_maximum][0] = 1

            try:
                match self.fit_type:
                    case constants.El.INDUCTOR:
                        f_upper_index = np.argwhere(np.logical_and(freq > res_fq, (magnitude_data) < (bw_value)))[0][0]
                    case constants.El.CAPACITOR:
                        f_upper_index = np.argwhere(np.logical_and(freq > res_fq, (magnitude_data) > (bw_value)))[0][0]
            except IndexError:
                #here we need to account for the fact that we could overshoot the max index
                if res_index + constants.DEFAULT_OFFSET_PEAK < len(freq):
                    f_upper_index = res_index + constants.DEFAULT_OFFSET_PEAK
                    bad_BW_flag[num_maximum][1] = 1
                else:
                    f_upper_index = len(freq) - 1

            # check if the found 3dB points are in an acceptable range i.e. not "behind" the next peak or "in front of"
            # the previous peak. If that is the case we set the index to a default offset to get a "bandwidth"
            if num_maximum != 0:
                if f_lower_index < mag_maxima_index[num_maximum - 1]:
                    f_lower_index = res_index - constants.DEFAULT_OFFSET_PEAK
                    bad_BW_flag[num_maximum] = 1
            if num_maximum < number_zones-1:
                if f_upper_index > mag_maxima_index[num_maximum + 1]:
                    #again we could overshoot the max index here
                    if res_index + constants.DEFAULT_OFFSET_PEAK < len(freq):
                        f_upper_index = res_index + constants.DEFAULT_OFFSET_PEAK
                        bad_BW_flag[num_maximum][1] = 1
                    else:
                        f_upper_index = len(freq) - 1
                        bad_BW_flag[num_maximum][1] = 1

            f_tuple = [freq[f_lower_index], res_fq, freq[f_upper_index]]
            bandwidth_list.append(f_tuple)
            peak_heights.append(abs(res_value))
            #THIS IS FOR TESTING
            if constants.DEBUG_BW_DETECTION:
                markerson = [f_lower_index,res_index,f_upper_index]
                plt.loglog(self.data_mag, '-bD', markevery=markerson)

        try:
            # here the last resonance can be "streched"; can be useful to "fill" the end zone of the curve
            # NOTE: the stretch factor is set to 1 most of the time, so there is no stretching happening
            stretch_factor = constants.BANDWIDTH_STRETCH_LAST_ZONE
            bandwidth_list[-1][2] = bandwidth_list[-1][2] * stretch_factor
            bandwidth_list[-1][0] = bandwidth_list[-1][0] * stretch_factor

        except IndexError:
            self.logger.info("INFO: No resonances found except the main resonance, consider a lower value for the prominence")

        self.peak_heights = peak_heights
        self.bandwidths = bandwidth_list
        self.bad_bandwidth_flag = bad_BW_flag
        self.order = len(self.bandwidths)

        return bandwidth_list

    def set_acoustic_resonance_frequency(self, res_fq):
        """
        Auxilliary method to set the acoustic resonance frequency for MLCCs

        :param res_fq: resonant frequency
        :return: None
        """
        self.acoustic_resonant_frequency = res_fq

    def fit_acoustic_resonance(self, param_set):
        """
        Method to fit the acoustic resonance of MLCCs


        :param param_set: a Parameters() object; should contain the parameters for the main resonance but not for higher order circuits

        :returns: a Parameters() object that is a copy of the Parameters() object passed to the function but containing
            the fitted circuit elements for the acoustic resonance (this Parameters() object is also stored in an instance variable)

        """
        freq = self.frequency_vector
        data = self.z21_data
        magnitude_data = self.data_mag
        f0 = self.f0

        # frequency limit the data to frequencies lower than the main resonance
        mag_data_lim = magnitude_data[freq<f0]
        freq_lim = freq[freq<f0]

        res_fq = self.acoustic_resonant_frequency
        res_index = np.argwhere(freq > res_fq)[0][0]

        res_value = data[res_index]
        bw_value = abs(res_value) * np.sqrt(2)

        # find upper 3dB point
        try:
            f_upper_index = np.argwhere(np.logical_and(freq > res_fq, (magnitude_data) > (bw_value)))[0][0]
            fu=freq[f_upper_index]
        except IndexError:
            # here we need to account for the fact that we could overshoot the max index
            if res_index + constants.DEFAULT_OFFSET_PEAK < len(freq):
                f_upper_index = res_index + int(constants.DEFAULT_OFFSET_PEAK / 2)
            else:
                f_upper_index = len(freq_lim) - 1

        # find lower 3dB point
        try:
            f_lower_index = np.flipud(np.argwhere((np.logical_and(freq < res_fq, (magnitude_data) > (bw_value)))))[0][0]
            fl = freq[f_lower_index]
        except IndexError:
            f_lower_index = res_index - int(constants.DEFAULT_OFFSET_PEAK / 2)

        freq_mdl = freq_lim[f_lower_index-10:f_upper_index+10]
        data_mdl = mag_data_lim[f_lower_index-10:f_upper_index+10]

        # use the bandwidth model to get estimates for the RLC circuit
        [bl,bu,R,L,C] = self.model_bandwidth(freq_mdl, data_mdl, res_fq)

        # correct the effect the main resonance has on the peak height
        main_res_here = self.calculate_Z(param_set, res_fq, 2, 0, 1, constants.fcnmode.OUTPUT)
        data_here = data[freq==res_fq]
        w_c = res_fq * 2 * np.pi
        Q = res_fq / (bu - bl)

        R_new = abs(1 / (1 / data_here[0] - 1 / main_res_here))
        C_new = 1/(R_new*w_c*Q)

        C=C_new
        params = copy.copy(param_set)

        #scale parameters to match units
        C = C / config.CAPUNIT
        L = L / config.INDUNIT
        w_c = w_c / config.FUNIT
        expr_string_L = '(1/((C_A*'+ str(config.CAPUNIT)+')*(w_A*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)

        #create the parameters for the acoustic resonance
        params.add('R_A', value=R_new, min=R*0.8, max=R_new*1.2, vary=True)
        params.add('C_A', value=C, min=C * 0.8, max=C * 1.5, vary=True)
        params.add('w_A', value =w_c, min = w_c*0.9, max=w_c*1.2, vary=True)
        params.add('L_A', value=L, expr=expr_string_L)


        #frequency limit the data to the bandwidth of the circuit and do a fit using the limited data
        modelfreq = freq[np.logical_and(freq > bl, freq < bu)]
        modeldata = data[np.logical_and(freq > bl, freq < bu)]

        out1 = minimize(self.calculate_Z, params,
                        args=(modelfreq, modeldata, 0, 0, constants.fcnmode.FIT,),
                        method='powell', options={'xtol': 1e-18, 'disp': True})

        # copy the parameters of the fit result
        params = copy.copy(out1.params)

        # finally set the parameters to not vary anymore
        self.change_parameter(params,'R_A',vary=False)
        self.change_parameter(params,'L_A',vary=False)
        self.change_parameter(params,'C_A',vary=False)
        self.change_parameter(params,'w_A',vary=False)

        # write to instance variable and return obtained parameters
        self.parameters = copy.copy(params)
        return params

    def get_acoustic_resonance(self):
        """
        Method to detect an acoustic resonance.
        Finds a peak in the data that is in front of (has lower frequency than) the main resonance.

        :return: obtained resonant frequency in Hz
        """
        freq = self.frequency_vector
        magnitude_data = self.data_mag
        f0 = self.f0

        # limit the data to before the main res
        mag_data_lim = magnitude_data[freq < f0]
        freq_lim = freq[freq < f0]

        mag_maxima = find_peaks(-20 * np.log10(mag_data_lim), height=-200, prominence=0)

        index_ac_res = np.argwhere(mag_maxima[1]['prominences'] == max(mag_maxima[1]['prominences']))[0][0]
        try:
            res_index = mag_maxima[0][index_ac_res]
            res_fq = freq_lim[res_index]
        except:
            res_fq = None

        return res_fq

    def calculate_hi_C_Rs(self):
        pass

    def create_hi_C_parameters(self, param_set):

        freq = self.frequency_vector

        C = self.nominal_value / config.CAPUNIT
        param_set.add('C', value = C, min = C*1e-1, max = C*1e1, vary = False)
        param_set.add('R_iso', value = constants.R_ISO_VALUE, min = constants.MIN_R_ISO, max = constants.MAX_R_ISO, vary = False)
        param_set.add('R_s', value = self.series_resistance, min = self.series_resistance*1e-1, max = self.series_resistance*1e1, vary = False)

        # if a resonance has been detected, calculate the value of the inductance of the main resonant circuit to not
        # get lower than 10 times the impedance of the first resonance
        if any(self.bandwidths):
            R1 = self.peak_heights[0]
            f1 = self.bandwidths[0][1]
            w1 = 2 * np.pi * f1
            Z_min = R1 * 1.0
            #values for the main resonance
            L_min = (Z_min/w1) / config.INDUNIT
            L_value = L_min * 1e1
            L_max = L_min * 1e2
            param_set.add('L', value = L_value, min = L_min, max = L_max)


            # find estimates for the first resonance

            stretch_factor = 1.5
            # get indices of the band
            b_l = self.bandwidths[0][0]
            b_u = self.bandwidths[0][2]

            f_l_index = np.argwhere(self.frequency_vector >= b_l)[0][0]
            f_u_index = np.argwhere(self.frequency_vector <= b_u)[-1][0]

            # calculate difference between upper and lower, so the number of points is relative to where we are in
            # the data, since the measurement points are not equally spaced
            # n_pts_offset = ((f_u_index - f_l_index) / 2) * stretch_factor

            # recalc lower and upper bound
            # f_l_index = f_c_index - int(np.floor(n_pts_offset))
            # f_u_index = f_c_index + int(np.floor(n_pts_offset))
            # get data for bandwidth model
            freq_BW_mdl = self.frequency_vector[f_l_index:f_u_index]
            data_BW_mdl = self.data_mag[f_l_index:f_u_index] * np.exp(1j * np.radians(self.data_ang[f_l_index:f_u_index]))

            [bl, bu, R1, L1, C1] = self.model_bandwidth(freq_BW_mdl,data_BW_mdl, f1)

            L1 = L1 / config.INDUNIT
            C1 = C1 / config.CAPUNIT
            w1 = w1 / config.FUNIT

            param_set.add('R1', value = R1, min = R1*0.5, max = R1*2, vary =True )
            param_set.add('C1', value=C1, min=C1 * 0.3, max=C1 * 3, vary = True)
            param_set.add('w1', value=w1, vary = False)
            param_set.add('BW1', value=((bu-bl))/config.FUNIT, vary = False)
            # param_set.add('L1', value=L1, min=L1 * 0.5, max=L1 * 2, vary = True)
            param_set.add('L1', expr = '(1/( '+ '((w1 * '+str(config.FUNIT)+') ** 2)' + '*(C1*' + str(config.CAPUNIT) + ')) )/'+str(config.INDUNIT)  )

            self.order = 1

            # # add another resonance as point of support between the bathtub model and the first resonance
            # f_A = f1/1.2
            # w_A = f_A*2*np.pi
            # Q_A = .8
            # R_A = 2 * self.series_resistance
            # C_A = 1 / (Q_A * R_A * w_A)
            # L_A = 1/((w_A**2)*C_A)
            #
            # C_A = C_A/config.CAPUNIT
            # L_A = L_A/config.INDUNIT
            # w_A = w_A/config.FUNIT
            # param_set.add('w_A', value = w_A, min = w_A*0.9, max = w_A*1.3)
            # param_set.add('R_A', value = R_A, min = R_A*0.1, max = R_A*1e1)
            # param_set.add('C_A', value = C_A, min = C_A*1e-1, max = C_A*1e1)
            # # param_set.add('L_A', value = L_A, min = L_A*1e-1, max = L_A*1e1)
            #
            # param_set.add('L_A', expr='(1/( ' + '((w_A * ' + str(config.FUNIT) + ') ** 2)' + '*(C_A*' + str(
            #     config.CAPUNIT) + ')) )/' + str(config.INDUNIT))


            #TODO: for testing purposes

            # f2 = f1/1.4
            # w2 = f2*2*np.pi
            # Q2 = Q_A
            # R2 = 2*self.series_resistance
            # C2 = 1/(Q2*R2*w2)
            # L2 = 1/((w2**2)*C2)
            #
            # C2 = C2 / config.CAPUNIT
            # L2 = L2 / config.INDUNIT
            # w2 = w2 / config.FUNIT
            # param_set.add('w2', value=w2, min=w2 * 0.9, max=w2 * 1.3)
            # param_set.add('BW2', value=0, vary = False)
            # param_set.add('R2', value=R2, min=R2 * 0.1, max=R2 * 1e1)
            # param_set.add('C2', value=C2, min=C2 * 1e-1, max=C2 * 1e1)
            # # param_set.add('L_A', value = L_A, min = L_A*1e-1, max = L_A*1e1)
            #
            # param_set.add('L2', expr='(1/( ' + '((w2 * ' + str(config.FUNIT) + ') ** 2)' + '*(C2*' + str(
            #     config.CAPUNIT) + ')) )/' + str(config.INDUNIT))
            #
            # f3 = f1 / 1.6
            # w3 = f3 * 2 * np.pi
            # Q3 = Q_A
            # R3 = 2 * self.series_resistance
            # C3 = 1 / (Q3 * R3 * w3)
            # L3 = 1 / ((w3 ** 2) * C3)
            #
            # C3 = C3 / config.CAPUNIT
            # L3 = L3 / config.INDUNIT
            # w3 = w3 / config.FUNIT
            # param_set.add('w3', value=w3, min=w3 * 0.9, max=w3 * 1.3)
            # param_set.add('BW3', value=0, vary=False)
            # param_set.add('R3', value=R3, min=R3 * 0.1, max=R3 * 1e1)
            # param_set.add('C3', value=C3, min=C3 * 1e-1, max=C3 * 1e1)
            # # param_set.add('L_A', value = L_A, min = L_A*1e-1, max = L_A*1e1)
            #
            # param_set.add('L3', expr='(1/( ' + '((w3 * ' + str(config.FUNIT) + ') ** 2)' + '*(C3*' + str(
            #     config.CAPUNIT) + ')) )/' + str(config.INDUNIT))

            # f4 = f1 / 1.8
            # w4 = f4 * 2 * np.pi
            # Q4 = Q_A
            # R4 = 2 * self.series_resistance
            # C4 = 1 / (Q4 * R4 * w4)
            # L4 = 1 / ((w4 ** 2) * C4)
            #
            # C4 = C4 / config.CAPUNIT
            # L4 = L4 / config.INDUNIT
            # w4 = w4 / config.FUNIT
            # param_set.add('w4', value=w4, min=w4 * 0.9, max=w4 * 1.3)
            # param_set.add('R4', value=R4, min=R4 * 0.1, max=R4 * 1e1)
            # param_set.add('C4', value=C4, min=C4 * 1e-1, max=C4 * 1e1)
            # # param_set.add('L_A', value = L_A, min = L_A*1e-1, max = L_A*1e1)
            #
            # param_set.add('L4', expr='(1/( ' + '((w3 * ' + str(config.FUNIT) + ') ** 2)' + '*(C3*' + str(
            #     config.CAPUNIT) + ')) )/' + str(config.INDUNIT))
            #
            # f3 = f1 / 2
            # w3 = f2 * 2 * np.pi
            # Q3 = Q_A
            # R3 = 2 * self.series_resistance
            # C3 = 1 / (Q3 * R3 * w3)
            # L3 = 1 / ((w3 ** 2) * C3)
            #
            # C3 = C3 / config.CAPUNIT
            # L3 = L3 / config.INDUNIT
            # w3 = w3 / config.FUNIT
            # param_set.add('w3', value=w3, min=w3 * 0.9, max=w3 * 1.3)
            # param_set.add('R3', value=R3, min=R3 * 0.1, max=R3 * 1e1)
            # param_set.add('C3', value=C3, min=C3 * 1e-1, max=C3 * 1e1)
            # # param_set.add('L_A', value = L_A, min = L_A*1e-1, max = L_A*1e1)
            #
            # param_set.add('L3', expr='(1/( ' + '((w3 * ' + str(config.FUNIT) + ') ** 2)' + '*(C3*' + str(
            #     config.CAPUNIT) + ')) )/' + str(config.INDUNIT))


            # self.order = 3





            #TODO: END for testing purposes




        #if there is no resonance present, we need to determine the value of the inductance via the slope of the dataset
        else:

            #TODO: detection of valid slope region could be bugged in cases where there is noisy data, this could use
            # a rework sometime

            slope_data = scipy.signal.savgol_filter(np.gradient(self.data_mag), 52, 3)
            maxslope_index = np.argwhere(slope_data == max(slope_data))[0][0]
            lr_offset = int(len(self.frequency_vector)*0.025)
            l_index = maxslope_index-lr_offset
            r_index = maxslope_index+lr_offset
            data = abs(self.z21_data[l_index:r_index])
            freq = self.frequency_vector[l_index:r_index]
            w = freq*2*np.pi
            L_values = data/w
            L_val = np.mean(L_values)/config.INDUNIT
            param_set.add('L', value=L_val, min=L_val*1e-1, max = L_val*1e1)





        # self.plot_curve(param_set, 0, 1)


        return param_set

    def fit_hi_C_model(self, param_set):
        freq = self.frequency_vector
        data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT_LOG

        modelfreq = freq[freq<=config.FREQ_UPPER_LIMIT]
        modeldata = data[freq<=config.FREQ_UPPER_LIMIT]

        if self.order:
            fit_main_resonance = 0
            out = minimize(self.calculate_Z, param_set,
                            args=(modelfreq, modeldata, fit_order, fit_main_resonance, mode,),
                            method='powell', options={'xtol': 1e-18, 'disp': True})
        else:
            fit_main_resonance = 1
            out = minimize(self.calculate_Z, param_set,
                           args=(modelfreq, modeldata, fit_order, fit_main_resonance, mode,),
                           method='powell', options={'xtol': 1e-18, 'disp': True})
        return out.params

    def create_nominal_parameters(self, param_set):
        """
        Function to create parameters for the elements of the main resonance.

        That is L, C, R_Fe and R_s for inductors and C, L, R_iso and R_s for capacitors

        :param param_set: A Parameters() object that the parameters will be written to
        :return: A Parameters() object containing the main resonance parameters
        """

        freq = self.frequency_vector
        res_value = abs(self.z21_data[self.f0_index])
        w0 = self.f0 * 2 * np.pi

        match self.fit_type:
            case constants.El.INDUCTOR:
                bw_value = res_value / np.sqrt(2)
                f_lower_index = np.flipud(np.argwhere(np.logical_and(freq < self.f0, self.data_mag < abs(bw_value))))[0][0]
                f_upper_index = (np.argwhere(np.logical_and(freq > self.f0, self.data_mag < abs(bw_value))))[0][0]
                BW = freq[f_upper_index] - freq[f_lower_index]
                R_Fe = (self.f0 * (self.f0 * 2 * np.pi) * self.nominal_value) / BW
                R_Fe = abs(self.z21_data[self.f0_index])
            case constants.El.CAPACITOR:
                #TODO: this is probably obsolete
                # bw_value = res_value * np.sqrt(2)
                # f_lower_index = np.flipud(np.argwhere(np.logical_and(freq < self.f0, self.data_mag > abs(bw_value))))[0][0]
                # f_upper_index = (np.argwhere(np.logical_and(freq > self.f0, self.data_mag > abs(bw_value))))[0][0]
                # BW = freq[f_upper_index] - freq[f_lower_index]
                R_Iso = constants.R_ISO_VALUE

        match self.fit_type:
            case constants.El.INDUCTOR:
                #calculate ideal capacitor for this resonance
                cap_ideal = 1 / (self.nominal_value * ((self.f0*2*np.pi) ** 2))


                #scale parameters to units used
                cap_ideal = cap_ideal / config.CAPUNIT
                w0 = w0 / config.FUNIT
                expression_string_L = '(1/((C*'+ str(config.CAPUNIT)+')*(w0*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)

                # add to parameters
                param_set.add('w0', value=w0, vary=False)
                param_set.add('R_Fe', value=R_Fe, min=constants.MIN_R_FE, max=constants.MAX_R_FE, vary=True)
                param_set.add('R_s', value=self.series_resistance, min=self.series_resistance * 0.1,
                                    max=self.series_resistance * 1.111, vary=True)
                param_set.add('C', value=cap_ideal,
                                    min=cap_ideal * constants.MAIN_RES_PARASITIC_LOWER_BOUND,
                                    max=cap_ideal * constants.MAIN_RES_PARASITIC_UPPER_BOUND, vary=True)
                param_set.add('L', expr = expression_string_L, vary=False)


            case constants.El.CAPACITOR:
                # calculate ideal inductor for this resonance
                ind_ideal = 1 / (self.nominal_value * ((self.f0 * 2 * np.pi) ** 2))

                #scale parameters to units used
                ind_ideal = ind_ideal / config.INDUNIT
                w0 = w0 / config.FUNIT
                expression_string_C = '(1/((L*' + str(config.INDUNIT) + ')*(w0*' + str(config.FUNIT) + ')**2))/' + str(config.CAPUNIT)

                param_set.add('w0', value=w0, vary=False)
                param_set.add('R_iso', value=R_Iso, min=constants.MIN_R_ISO, max=constants.MAX_R_ISO, vary=True)
                param_set.add('R_s', value=self.series_resistance, min=self.series_resistance * 0.01,
                                    max=self.series_resistance * 1.111, vary=False)
                param_set.add('L', value=ind_ideal,
                                    min=ind_ideal * constants.MAIN_RES_PARASITIC_LOWER_BOUND,
                                    max=ind_ideal * constants.MAIN_RES_PARASITIC_UPPER_BOUND, vary=True)
                param_set.add('C', expr = expression_string_C, vary = False)



        return param_set

    def create_higher_order_parameters(self, config_number, param_set):
        """
        Method to create the circuit elements for the higher order resonances.
        Can create parameters in two configurations.

        GUI_config 1 constrains one of the energy storing elements by the Q-factor and the other is bound by the resonant
        frequency, leaving only the center frequency and the resistor adjustable.
        GUI_config 2 only constrains one energy storing element to the other by resonant frequency. This is the configuration
        known to perform better if the main resonance fit is fairly accurate.

        :param config_number: selecting the configuration of the resulting parameters. Can be either 1 or 2.
        :param param_set: A Parameters() object that the circuit elements will be written to
        :return: A Parameters() object containing the higher order circuit elements
        """

        #if we got too many frequency zones -> restrict fit to max order
        #else get order from frequency zones and write found order to class
        if constants.MAX_ORDER >= len(self.bandwidths):
            order = len(self.bandwidths)
            self.order = len(self.bandwidths)
        else:
            order = constants.MAX_ORDER
            self.order = order
            self.logger.info("Info: more resonances detected than maximum order permits, set order to {value}".format(value=order))
            #TODO: some methods are not robust enough for this fit maybe?

        freq = self.frequency_vector
        data = self.z21_data
        self.modeled_bandwidths = np.zeros([self.order, 3])
        main_res_data = self.calculate_Z(param_set, self.frequency_vector, 2, 0, 0,
                                         constants.fcnmode.OUTPUT)


        for key_number in range(1, order + 1):

            #create keys
            C_key   = "C%s" % key_number
            L_key   = "L%s" % key_number
            R_key   = "R%s" % key_number
            w_key   = "w%s" % key_number
            BW_key  = "BW%s" % key_number




            #get upper and lower frequencies
            b_l = self.bandwidths[key_number - 1][0]
            b_c = self.bandwidths[key_number - 1][1]
            b_u = self.bandwidths[key_number - 1][2]

            # handle bandwidths here -> since the bandwidth detection relies on the 3dB points, which are not always
            # present, we may need to "model" the BW. If we have one of the two 3dB points though, we can assume symmetric
            # bandwidth EDIT: bandwidth model has been applied to all peaks, since it gives good estimates for the
            # parameter values

            stretch_factor = 1.5
            #get indices of the band
            f_c_index = np.argwhere(self.frequency_vector == b_c)[0][0]
            f_l_index = np.argwhere(self.frequency_vector == b_l)[0][0]
            f_u_index = np.argwhere(self.frequency_vector == b_u)[0][0]
            #calculate diffference between upper and lower, so the number of points is relative to where we are in
            #the data, since the measurement points are not equally spaced
            n_pts_offset = ((f_u_index - f_l_index) / 2) * stretch_factor
            #recalc lower and upper bound
            f_l_index = f_c_index - int(np.floor(n_pts_offset))
            f_u_index = f_c_index + int(np.floor(n_pts_offset))
            #get data for bandwidth model
            freq_BW_mdl = self.frequency_vector[f_l_index:f_u_index]
            data_BW_mdl = self.data_mag[f_l_index:f_u_index]*np.exp(1j*np.radians(self.data_ang[f_l_index:f_u_index]))

            #upper and lower 3dB point faulty
            if self.bad_bandwidth_flag[key_number-1].all:
                #now model the BW
                [b_l,b_u,r_value,value_ind,value_cap] = self.model_bandwidth(freq_BW_mdl,data_BW_mdl,b_c)
            #only lower 3dB point faulty
            elif self.bad_bandwidth_flag[key_number-1][0]:
                [_,_, r_value, value_ind, value_cap] = self.model_bandwidth(freq_BW_mdl, data_BW_mdl, b_c)
                b_l = b_c - (b_u - b_c)
            # only upper 3dB point faulty
            elif self.bad_bandwidth_flag[key_number - 1][1]:
                [_,_, r_value, value_ind, value_cap] = self.model_bandwidth(freq_BW_mdl, data_BW_mdl, b_c)
                b_u = b_c + (b_c - b_l)
            #both points present -> we only want estimates for the elements
            else:
                [_,_, r_value, value_ind, value_cap] = self.model_bandwidth(freq_BW_mdl, data_BW_mdl, b_c)


            # bandwidth
            BW_min = (b_u - b_l) * constants.BW_MIN_FACTOR
            BW_max = (b_u - b_l) * constants.BW_MAX_FACTOR
            BW_value = (b_u - b_l)  # BW_max / 8

            #rewrite the obtained bandwidth
            self.modeled_bandwidths[key_number - 1][0] = b_l
            self.modeled_bandwidths[key_number - 1][1] = b_c
            self.modeled_bandwidths[key_number - 1][2] = b_u

            # center frequency (omega)
            w_c = b_c * 2 * np.pi
            min_w = w_c * constants.MIN_W_FACTOR
            max_w = w_c * constants.MAX_W_FACTOR

            ############################# PRE-Fit ######################################################################

            if self.fit_type == constants.El.CAPACITOR:

                #calculate the
                curve_data = self.calculate_Z(param_set, freq, 2, key_number-1, 0,constants.fcnmode.OUTPUT)
                data_here = data[freq == b_c]
                main_res_here = curve_data[freq == b_c]

                w_c = b_c * 2 * np.pi
                Q = b_c / (b_u - b_l)


                R_adjusted = abs(1 / (1 / data_here[0] - 1 / main_res_here[0]))
                C_adjusted = 1 / (R_adjusted * w_c * Q)

                r_value = R_adjusted
                value_cap = C_adjusted



            if self.fit_type == constants.El.INDUCTOR:

                # calculate the
                curve_data = self.calculate_Z(param_set, freq, 2, key_number - 1, 0,
                                              constants.fcnmode.OUTPUT)
                data_here = data[freq == b_c]
                main_res_here = curve_data[freq == b_c]

                w_c = b_c * 2 * np.pi
                Q = b_c / (b_u - b_l)

                # adjust the value of R; since the BW model provides us with an R for the "standalone" circuit, we need
                # to correct it to account for the main res data as well
                R_adjusted = abs( abs(data_here[0]) -  abs(main_res_here[0]) )

                C_adjusted = Q / (R_adjusted * w_c)

                r_value = R_adjusted
                value_cap = C_adjusted


            # rescale parameters to match units
            value_cap = value_cap / config.CAPUNIT

            w_c = w_c / config.FUNIT
            min_w = min_w / config.FUNIT
            max_w = max_w / config.FUNIT
            BW_value = BW_value / config.FUNIT
            BW_min = BW_min / config.FUNIT
            BW_max = BW_max / config.FUNIT




            #################### CAPACITORS ############################################################################
            if self.fit_type == constants.El.CAPACITOR:

                # good values for capacitor fitting
                max_cap = value_cap * 1e1
                min_cap = value_cap * 1e-1

                r_max = r_value * 1.01
                r_min = r_value * 0.990

                expression_string_C = '(('+BW_key +'*'+ str(config.FUNIT) +'*'+ str(2*np.pi)+')/(('+w_key+'*'+str(config.FUNIT)+')**2*'+R_key+'))/'+str(config.CAPUNIT)



                param_set.add(BW_key, min=BW_min, max=BW_max, value=BW_value, vary=False)
                match config_number:
                    case 1:

                        #TODO: check expression string for case 1 with unit scaling
                        expression_string_L = '(1/(('+ C_key +'*'+ str(config.CAPUNIT)+')*('+w_key+'*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)

                        param_set.add(w_key, min=min_w, max=max_w, value=w_c, vary=True)
                        param_set.add(R_key, value = r_value, max = r_max, min = r_min, vary = True)
                        param_set.add(C_key, expr=expression_string_C, vary=False)
                        param_set.add(L_key, expr=expression_string_L, vary=False)
                    case 2:
                        expression_string_L = '(1/(('+ C_key +'*'+ str(config.CAPUNIT)+')*('+w_key+'*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)

                        param_set.add(C_key, min=min_cap, max=max_cap, value=value_cap, vary=True)
                        param_set.add(R_key, value=r_value, min=r_value * 0.2, max=r_value * 5)
                        param_set.add(w_key, min=min_w, max=max_w, value=w_c, vary=True)
                        param_set.add(L_key, expr=expression_string_L)



            #################### INDUCTORS #############################################################################
            else:

                max_cap = value_cap * 1e1#2
                min_cap = value_cap * 1e-1#500e-3


                param_set.add(BW_key, min=BW_min, max=BW_max, value=BW_value, vary=False)
                match config_number:
                    case 1:
                        r_min= r_value*0.9
                        r_max= r_value*1.1
                        #GUI_config B -> default GUI_config; this goes via the Q factor
                        expression_string_L = '(1/(('+ C_key +'*'+ str(config.CAPUNIT)+')*('+w_key+'*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)
                        expression_string_C = '(1/(' +str(2*np.pi)+'*'+BW_key+'*'+str(config.FUNIT)+'*'+R_key +'))/'+str(config.CAPUNIT)

                        param_set.add(w_key, min=min_w, max=max_w, value=w_c, vary=True)
                        param_set.add(R_key, value=r_value, min = r_min, max=r_max, vary=True)
                        param_set.add(C_key, expr=expression_string_C, vary=False)
                        param_set.add(L_key, expr=expression_string_L, vary=False)

                    case 2:

                        expression_string_L = '(1/(('+ C_key +'*'+ str(config.CAPUNIT)+')*('+w_key+'*'+str(config.FUNIT)+')**2))/'+str(config.INDUNIT)

                        param_set.add(C_key, min=min_cap, max=max_cap, value=value_cap, vary=True)
                        param_set.add(R_key, value = r_value, min = r_value * 0.5, max = r_value * 2)
                        param_set.add(w_key, min=min_w, max=max_w, value=w_c, vary=True)
                        param_set.add(L_key, expr=expression_string_L)


        return param_set

    def pre_fit_bands(self, param_set):
        """
        Method to fit the resonant circuits one-by-one to the impedance data

        The resonant circuits are stepped through. The resonant circuit selected is freed, letting its parameters vary
        while the other parameters are fixed in place. After the fit has completed, the circuits parameters are constrained
        again to not vary and the next circuit will be fit.
        This improves the overall accuracy of the model.

        :param param_set: A Parameters() object containing the resonant circuits
        :return: A Parameters() object containing the fitted bands
        """

        #first fix all parameters in place
        self.fix_parameters(param_set)

        #initiate variables needed
        freq = self.frequency_vector
        data = self.z21_data
        # self.plot_curve(self.parameters, self.order, 0)

        #step through all the bands and 'cut' the frequency range of the resonant circuit
        for it, band in enumerate(self.modeled_bandwidths):
            fit_freq = freq[np.logical_and(freq > band[0], freq < band[2])]
            fit_data = data[np.logical_and(freq > band[0], freq < band[2])]
            # generate keys for the parameters we want to vary
            key_number = it+1
            R_key = 'R%s' % key_number
            L_key = 'L%s' % key_number
            C_key = 'C%s' % key_number
            w_key = 'w%s' % key_number
            #set the parameters in question to vary
            self.change_parameter(param_set, R_key, vary = True)
            self.change_parameter(param_set, C_key, vary = True)
            self.change_parameter(param_set, L_key, vary = True)
            self.change_parameter(param_set, w_key, vary = True)

            #do the fit
            out = minimize(self.calculate_Z, param_set,
                                args=(fit_freq, fit_data, self.order, 0, FIT_BY,),
                                method='powell', options={'xtol': 1e-18, 'disp': True})

            #write fit results to parameters and set their 'vary' to False
            R = out.params[R_key].value
            L = out.params[L_key].value
            C = out.params[C_key].value
            w = out.params[w_key].value
            self.change_parameter(param_set, R_key, value = R, min = R*0.9, max = R*1.1, vary=False)
            self.change_parameter(param_set, L_key, value = L, min = L*0.9, max = L*1.1,vary=False)
            self.change_parameter(param_set, C_key, value = C, min = C*0.9, max = C*1.1,vary=False)
            self.change_parameter(param_set, w_key, value = w, min = w*0.9, max = w*1.1,vary=False)

        #free parameters for further fitting
        self.free_parameters_higher_order(param_set)
        return param_set

    def correct_parameters(self, param_set, change_main, num_it = 2):
        """
        Method to correct the parameters of the set. Corrects higher order resonances, but can also correct the main
        resonance parameters if needed.

        When the circuit elements are created, their resistance value is roughly the magnitude at resonance minus the
        magnitude of the main resonance at that point. However as new elements are introduced into the model, the peak
        values of the resonances slightly change, since every circuit influences all other circuits. This method aims to
        correct the resistance values and (inherently to keep the same Q-factor for the circuit) the capacitor.

        Note: does not work for the acoustic resonance of MLCCs

        :param param_set: A Parameters() object containing the higher order circuits and main resonance
        :param change_main: Boolean. Selects whether to correct the main resonance parameters
        :param num_it: Number of iterations of the correction (is 2 by default)
        :return: A Parameters() object containing the corrected parameters

        """

        freq = self.frequency_vector
        order = self.order
        data = self.z21_data
        params = copy.copy(param_set)



        for it in range(num_it):
            #at the start of each iteration correct main resonance
            curve_data = self.calculate_Z(params, freq, 2, order, 0, constants.fcnmode.OUTPUT)
            if self.fit_type == constants.El.INDUCTOR and change_main:
                # get Q value
                b_l = np.flipud(np.argwhere(np.logical_and(freq < self.f0, abs(data) < abs(data[freq == self.f0][0])/(np.sqrt(2)))))[0][0]
                b_u = np.argwhere(np.logical_and(freq > self.f0, abs(data) < abs(data[freq == self.f0][0])/(np.sqrt(2)) ))[0][0]
                w0 = (self.f0*2*np.pi)
                Q_main = self.f0/(freq[b_u]-freq[b_l])
                #adjust R_Fe by difference from the data
                R_diff = abs(data[freq == self.f0][0]) - params['R_Fe'].value
                R_new = params['R_Fe'].value + R_diff
                #adjust C depending on Q and new R_Fe
                C_new = Q_main / (w0*R_new)
                #C_new needs to be rescaled to match units
                C_new = C_new / config.CAPUNIT
                self.change_parameter(params, 'R_Fe', min = R_new *0.8, max = R_new*1.2, value = R_new, vary = False)
                self.change_parameter(params, 'C', min = C_new*0.8, max = C_new*1.2, value= C_new, vary = False)

            elif self.fit_type == constants.El.CAPACITOR and change_main:
                # get Q value
                b_l = np.flipud(np.argwhere(np.logical_and(freq < self.f0, abs(data) > abs(data[freq == self.f0][0]) * (np.sqrt(2)))))[0][0]
                b_u = np.argwhere(np.logical_and(freq > self.f0, abs(data) > abs(data[freq == self.f0][0]) * (np.sqrt(2))))[0][0]
                w0 = (self.f0 * 2 * np.pi)
                Q_main = self.f0 / (freq[b_u] - freq[b_l])
                # adjust R_s by difference from the data
                R_diff = abs(data[freq == self.f0][0]) - params['R_s'].value
                R_new = params['R_s'].value + R_diff
                # adjust L depending on Q and new R_Fe
                L_new = (R_new*Q_main)/w0
                #rescale L_new to match units
                L_new = L_new / config.INDUNIT
                self.change_parameter(params, 'R_s', min=R_new * 0.8, max=R_new * 1.2, value=R_new, vary=False)
                self.change_parameter(params, 'L', min=L_new * 0.8, max=L_new * 1.2, value=L_new, vary=False)


            for key_number in range(1, order + 1):
                index = key_number - 1
                if self.fit_type == constants.El.INDUCTOR:
                    curve_data = self.calculate_Z(params, freq, 2, order, 0, constants.fcnmode.OUTPUT)
                    band = self.bandwidths[index]
                    dataindex = np.argwhere(freq == band[1])[0][0]
                    #check if parameter needs to be corrected -> 5% relative errror is the metric
                    if abs(abs(data[dataindex]) - abs(curve_data[dataindex])) / abs(data[dataindex]) >= 0.05:

                        R_key = 'R%s' % key_number
                        C_key = 'C%s' % key_number

                        #to get a better estimate for the resistive value, look at the real part and find the peak
                        try:
                            data_lim = self.z21_data[np.logical_and(freq < band[2], freq > band[0])]
                            peak = find_peaks(np.real(data_lim), height = 0)
                            match self.fit_type:
                                case constants.El.INDUCTOR:
                                    r_peak_index = np.argwhere(peak[1]['peak_heights'] == max(peak[1]['peak_heights']))[0][0]
                                    r_peak = peak[1]['peak_heights'][r_peak_index]
                                case constants.El.CAPACITOR:
                                    r_peak_index = np.argwhere(peak[1]['peak_heights'] == min(peak[1]['peak_heights']))[0][0]
                                    r_peak = peak[1]['peak_heights'][r_peak_index]
                        except:
                            r_peak = np.real(data[dataindex])

                        R = params[R_key].value
                        R_diff = r_peak - np.real(curve_data[dataindex])
                        if (R + R_diff) > 0:
                            R_adjusted = R + R_diff
                            w_c = params['w%s' % key_number].value * config.FUNIT
                            BW = params['BW%s' % key_number].value * config.FUNIT
                            Q = w_c / (BW*2*np.pi)
                            C_adjusted = Q / (R_adjusted * w_c)
                            #rescale parameter to match units
                            C_adjusted = C_adjusted / config.CAPUNIT
                            self.change_parameter(params, R_key, min = R_adjusted*0.2, max = R_adjusted *5, value = R_adjusted, vary = True, expr ='')
                            self.change_parameter(params, C_key, min = C_adjusted*1e-1, max = C_adjusted *1e1, value = C_adjusted, vary = True)
                        else:
                            #if we can't find a valid correction, leave it be
                            self.logger.info('Parameter not corrected: ' + R_key + ' ;run: ' + self.file.name)



        return params

    def calculate_Z(self, parameters, frequency_vector, data, fit_order, fit_main_res, modeflag):
        """
        Objective function that is invoked by the optimizer. Calculates the impedance for the model.

        This calculates the impedance for the main resonance and then steps through all the higher order resonances in
        order to calculate the impedance for the whole model.

        Has different output modes:

        - Impedance (OUTPUT)
        - Difference in magnitude for the given data (FIT)
        - Difference in phase for the given data (ANGLE)
        - Difference in real part for the given data (FIT_REAL)
        - Difference in imag part for the given data (FIT_IMAG)

        Note: this **does** account for the acoustic resonance of MLCCs.

        :param parameters: A Parameters() object containing the parameters of the model
        :param frequency_vector: The frequency vector over which the impedance is requested.
        :param data: The data to be used for all kinds of FIT output
        :param fit_order: The order of the model (i.e. the number of resonant circuits)
        :param fit_main_res: Boolean. Decides if only the main resonance shall be fit (TRUE) or if higher order
                resonances shall be calculated too (FALSE)
        :param modeflag: Decides which output the function shall give. Can be FIT, OUTPUT, ANGLE, FIT_REAL or FIT_IMAG
        :return: Either the impedance of the model or a difference, depends on modeflag
        """

        #if we only want to fit the main resonant circuit, set order to zero to avoid "for" loops
        if fit_main_res:
            order = 0
        else:
            order = fit_order

        #create array for frequency
        freq = frequency_vector
        w = freq * 2 * np.pi

        #get parameters for main circuit
        C = parameters['C'].value * config.CAPUNIT
        L = parameters['L'].value * config.INDUNIT
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

        #if MLCC
        if self.captype == constants.captype.MLCC and not fit_main_res:
            Z_A = parameters['R_A'].value + (1j*w*parameters['L_A'].value*config.INDUNIT) + (1/(1j*w*parameters['C_A'].value*config.CAPUNIT))
            Z = 1/(1/Z_main + 1/Z_A)


        for actual in range(1, order + 1):
            key_number = actual
            C_key = "C%s" % key_number
            L_key = "L%s" % key_number
            R_key = "R%s" % key_number
            C_act = parameters[C_key].value * config.CAPUNIT
            L_act = parameters[L_key].value * config.INDUNIT
            R_act = parameters[R_key].value
            Z_C   = 1 / (1j * w * C_act)
            Z_L   = (1j * w * L_act)
            Z_R   = R_act
            match self.fit_type:
                case constants.El.INDUCTOR:
                    Z    += 1 / ( (1/Z_C) + (1/Z_L) + (1/Z_R) )
                case constants.El.CAPACITOR:
                    Z = 1 / ( 1/Z + 1/(Z_R + Z_L + Z_C))

        match modeflag:
            case fcnmode.FIT:
                diff = abs(data)-abs(Z)
                return (diff)
            case fcnmode.FIT_LOG:
                diff = np.log10(abs(data)) - np.log10(abs(Z))
                return (diff)
            case fcnmode.OUTPUT:
                return Z
            case fcnmode.ANGLE:
                return np.angle(data)-np.angle(Z)
            case fcnmode.FIT_REAL:
                return np.real(data)-np.real(Z)
            case fcnmode.FIT_IMAG:
                return np.imag(data)-np.imag(Z)

    def fit_curve_higher_order(self, param_set):
        """
        Method to fit all higher order resonances.

        :param param_set: A Parameters() object containing the circuits of the model
        :return: A Parameters() object containing the fitted circuits
        """
        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_order = self.order

        # Frequency limit data for fit
        fit_data_frq_lim = fit_data[np.logical_and(freq > self.f0 * constants.MIN_ZONE_OFFSET_FACTOR,
                                                   freq < config.FREQ_UPPER_LIMIT)]
        freq_data_frq_lim = freq[np.logical_and(freq > self.f0 * constants.MIN_ZONE_OFFSET_FACTOR,
                                                freq < config.FREQ_UPPER_LIMIT)]

        # fit the parameter set
        if self.order:
            fit_main_resonance = 0
            out = minimize(self.calculate_Z, param_set,
                            args=(freq_data_frq_lim, fit_data_frq_lim, fit_order, fit_main_resonance, FIT_BY,),
                            method='powell', options={'xtol': 1e-18, 'disp': True})
            return out.params

        else:
            return param_set

    def fit_main_res_inductor_file_1(self, param_set):
        """
        Method to fit the main resonance circuit for an inductor for the first file (i.e. the reference file)

        :param param_set: A Parameters() object containing the parameters for the main resonance circuit
        :return: A Parameters() object containing the fitted parameters of the main resonance
        """

        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT

        if constants.DEBUG_FIT: #debug plot -> main res before fit
            self.plot_curve(param_set, 0, 1)

        # frequency limit data (upper bound) so there are (ideally) no higher order resonances in the main res fit data
        fit_main_resonance = 1
        freq_for_fit = freq[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]
        data_for_fit = fit_data[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]

        # crop some samples of the start of data (~100) because the slope at the start of the dataset might be off
        freq_for_fit = freq_for_fit[self.offset:]
        data_for_fit = data_for_fit[self.offset:]

        #################### Main Resonance ############################################################################

        # start by fitting the main res with all parameters set to vary
        out = minimize(self.calculate_Z, param_set,
                            args=(freq_for_fit, data_for_fit, fit_order, fit_main_resonance, mode,),
                            method='powell', options={'xtol': 1e-18, 'disp': True})

        # set all parameters to not vary; let only R_s vary
        for pname, par in out.params.items():
            par.vary = False
        out.params['R_s'].vary = True

        # fit R_s via the phase of the data
        out = minimize(self.calculate_Z, out.params,
                       args=(
                           freq_for_fit, data_for_fit, fit_order, fit_main_resonance, constants.fcnmode.ANGLE,),
                       method='powell', options={'xtol': 1e-18, 'disp': True})

        # fitting R_s again does change the main res fit, so set L an C to vary
        out.params['R_s'].vary = False
        out.params['C'].vary = True
        out.params['L'].vary = True

        # and fit again
        out = minimize(self.calculate_Z, out.params,
                            args=(freq_for_fit, data_for_fit, fit_order, fit_main_resonance, mode,),
                            method='powell', options={'xtol': 1e-18, 'disp': True})

        # write series resistance to class variable (important if other files are fit)
        self.series_resistance = out.params['R_s'].value

        #set parameters of self to fitted main resonance
        self.parameters = out.params
        param_set = out.params

        # fix main resonance parameters in place
        self.fix_main_resonance_parameters(param_set)

        if constants.DEBUG_FIT:  # debug plot -> fitted main resonance
            self.plot_curve(param_set, 0, 1)

        return param_set

    def fit_main_res_capacitor_file_1(self, param_set):
        """
        Method to fit the main resonance circuit for a capacitor for the first file (i.e. the reference file)

        :param param_set: A Parameters() object containing the parameters for the main resonance circuit
        :return: A Parameters() object containing the fitted parameters of the main resonance
        """

        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT
        if self.captype == constants.captype.HIGH_C:
            mode = constants.fcnmode.FIT_LOG

        if constants.DEBUG_FIT:  # debug plot -> main res before fit
            self.plot_curve(param_set, 0, 1)

        ###################### Main resonance ##########################################################################

        # frequency limit data (upper bound) so there are (ideally) no higher order resonances in the main res fit data
        fit_main_resonance = 1

        freq_for_fit = freq[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]
        data_for_fit = fit_data[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]

        # crop some samples of the start of data (~100) because the slope at the start of the dataset might be off
        if self.offset > 0:
            offset = self.offset
        else:
            offset = 50

        freq_for_fit = freq_for_fit[offset:]
        data_for_fit = data_for_fit[offset:]

        if captype != constants.captype.HIGH_C:
            param_set['R_s'].vary = False

        out = minimize(self.calculate_Z, param_set,
                       args=(freq_for_fit, data_for_fit, fit_order, fit_main_resonance, mode,),
                       method='powell', options={'xtol': 1e-18, 'disp': True})

        # create datasets for data before/after fit
        old_data = self.calculate_Z(param_set, freq_for_fit, [], 0, fit_main_resonance,
                                    constants.fcnmode.OUTPUT)
        new_data = self.calculate_Z(out.params, freq_for_fit, [], 0, fit_main_resonance,
                                    constants.fcnmode.OUTPUT)
        data_frq_lim = data_for_fit  # self.z21_data[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)][self.offset:]

        # check if the main resonance fit yields good results -> else: go with initial guess
        if abs(sum(np.log10(abs(new_data)) - np.log10(abs(data_frq_lim)))) < abs(sum(np.log10(abs(old_data)) - np.log10(abs(data_frq_lim)))):
            if constants.DEBUG_MESSAGES:
                self.logger.info("CAPFIT MR: did fit")
            param_set = out.params
        else:
            if constants.DEBUG_MESSAGES:
                self.logger.info("CAPFIT MR: did not fit (used param set as is)")
            # redundant, but for readability
            param_set = param_set

            # this is kind of a hotfix; if there are no higher order resonances present, out.params is not overwritten
            # so the "worse" param GUI_config is taken in the final result
            out.params = param_set

        # fix main resonance parameters in place
        self.fix_main_resonance_parameters(param_set)

        if constants.DEBUG_FIT:  # debug plot -> fitted main resonance
            self.plot_curve(param_set, 0, 1)

        return param_set

    def fit_main_res_inductor_file_n(self, param_set, debug_plots=False):
        """
        Method to fit the main resonance circuit for an inductor for a file that is not the reference file

        :param param_set: A Parameters() object containing the parameters for the main resonance circuit
        :return: A Parameters() object containing the fitted parameters of the main resonance
        """

        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT

        if debug_plots:  # debug plot -> main res before fit
            self.plot_curve(param_set, 0, 1, str(self.file.name) + 'MR before fit')

        # frequency limit data (upper bound) so there are (ideally) no higher order resonances in the main res fit data
        fit_main_resonance = 1
        freq_for_fit = freq[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]
        data_for_fit = fit_data[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]

        # crop some samples of the start of data (~100) because the slope at the start of the dataset might be off
        freq_for_fit = freq_for_fit[self.offset:]
        data_for_fit = data_for_fit[self.offset:]

        #################### Main Resonance ############################################################################

        # start by fitting the main res with all parameters set to vary
        out = minimize(self.calculate_Z, param_set,
                       args=(freq_for_fit, data_for_fit, fit_order, fit_main_resonance, mode,),
                       method='powell', options={'xtol': 1e-18, 'disp': True})

        # set parameters of self to fitted main resonance (should be obsolete though)
        self.parameters = out.params
        param_set = out.params

        # fix main resonance parameters in place
        self.fix_main_resonance_parameters(param_set)

        if debug_plots:  # debug plot -> fitted main resonance
            self.plot_curve(param_set, 0, 1, str(self.file.name) + 'MR after fit')

        return param_set

    def fit_main_res_capacitor_file_n(self, param_set, debug_plots=False):
        """
        Method to fit the main resonance circuit for a capacitor for a file that is not the reference file

        :param param_set: A Parameters() object containing the parameters for the main resonance circuit
        :return: A Parameters() object containing the fitted parameters of the main resonance
        """

        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT

        if debug_plots:  # debug plot -> main res before fit
            self.plot_curve(param_set, 0, 1, str(self.file.name) + 'MR before fit')

        # frequency limit data (upper bound) so there are (ideally) no higher order resonances in the main res fit data
        fit_main_resonance = 1
        freq_for_fit = freq[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]
        data_for_fit = fit_data[(freq < self.f0 * constants.MIN_ZONE_OFFSET_FACTOR)]

        # crop some samples of the start of data (~100) because the slope at the start of the dataset might be off
        freq_for_fit = freq_for_fit[self.offset:]
        data_for_fit = data_for_fit[self.offset:]

        #################### Main Resonance ############################################################################

        # start by fitting the main res with all parameters set to vary
        out = minimize(self.calculate_Z, param_set,
                       args=(freq_for_fit, data_for_fit, fit_order, fit_main_resonance, mode,),
                       method='powell', options={'xtol': 1e-18, 'disp': True})

        # set parameters of self to fitted main resonance (should be obsolete though)
        self.parameters = out.params
        param_set = out.params

        # fix main resonance parameters in place
        self.fix_main_resonance_parameters(param_set)

        if debug_plots:  # debug plot -> fitted main resonance
            self.plot_curve(param_set, 0, 1, str(self.file.name) + 'MR before fit')

        return param_set

    def select_param_set(self, params: list):
        """
        A method to select the parameter set that fits "better" if multiple parameter sets have been created (that is the
        case if there are multiple configurations)

        This function calls the calculate_band_norm() function which returns a metric of how well a certain parameter set
        fits the bands. That means the parameter sets are tested on how well they fit the corresponding resonances in the
        data.

        :param params: A list of Parameters() objects for the different models to test
        :param debug: Boolean. If True the function logs which set it took
        :return: The selected Parameters() object
        """
        freq = self.frequency_vector
        fit_data = self.z21_data
        fit_main_resonance = False
        order = self.order
        mode = constants.fcnmode.OUTPUT

        model_data = []
        norm = []
        for it, param_set in enumerate(params):
            model_data.append(self.calculate_Z(param_set, freq, [], order, fit_main_resonance, mode))
            norm.append(self.calculate_band_norm(model_data[it]))

        least_norm_mdl_index = np.argwhere(norm == min(norm))[0][0]

        if constants.DEBUG_MESSAGES:
            self.logger.info(self.file.name + ": selected parameter set " + str(least_norm_mdl_index + 1))


        return params[least_norm_mdl_index]

    def overwrite_main_res_params_file_n(self, param_set, param_set0):
        """
        Function to overwrite and recalculate the main resonance parameters for a file that is not the reference file.

        :param param_set: A Parameters() object containing the main resonance parameters for the given file
        :param param_set0: A Parameters() object containing the main resonance parameters for the reference file
        :return: A Parameters() object containing the updated main resonance paramters
        """
        fit_data = self.z21_data
        freq = self.frequency_vector

        match self.fit_type:
            case constants.El.INDUCTOR:
                L_ideal = 1 / ((self.f0 * 2 * np.pi) ** 2 * (param_set0['C'].value * config.CAPUNIT))
                self.nominal_value = L_ideal
                L_ideal = L_ideal / config.INDUNIT
                C_val = param_set0['C'].value
                R_s   = param_set0['R_s'].value
                self.change_parameter(param_set, param_name='C', value=C_val, min=C_val * 0.8, max=C_val * 1.2,
                                      vary=False)
                self.change_parameter(param_set, param_name='R_s', value=R_s, min=R_s * 0.8, max=R_s * 1.2,
                                      vary=False)
                self.change_parameter(param_set, param_name='L', value=L_ideal, vary=True, min=L_ideal * 0.8,
                                      max=L_ideal * 1.25, expr='')

            case constants.El.CAPACITOR:
                C_ideal = 1 / ((self.f0 * 2 * np.pi) ** 2 * (param_set0['L'].value * config.INDUNIT))
                self.nominal_value = C_ideal
                C_ideal = C_ideal/config.CAPUNIT
                L_val = param_set0['L'].value
                R_iso = param_set0['R_iso'].value
                # write back value for L and R_iso
                self.change_parameter(param_set, param_name='L', value=L_val, min=L_val * 0.8, max=L_val * 1.2,
                                      vary=False)
                self.change_parameter(param_set, param_name='R_iso', value=R_iso, vary=False)
                # get new value for R_s
                R_s = abs(fit_data[freq == self.f0][0])
                self.change_parameter(param_set, param_name='R_s', value=R_s, vary=False)
                self.change_parameter(param_set, param_name='C', value=C_ideal, vary=True, min=C_ideal * 0.999,
                                      max=C_ideal * 1.001, expr='')
        self.parameters = param_set
        return param_set


    ####################################V AUXILLIARY V##################################################################
    def write_model_data(self, param_set, model_order):
        """
        Auxilliary function to calculate the impedance of the model

        :param param_set: A Parameters() object containing the circuits of the model
        :param model_order: The order of the model i.e. the number of resonant circuits
        :return: None (stores the model data in an instance variable)
        """
        freq = self.frequency_vector
        mode = constants.fcnmode.OUTPUT
        order = model_order
        fit_main_resonance = 0
        self.model_data = self.calculate_Z(param_set, freq, [], order, fit_main_resonance, mode)

    def change_parameter(self, param_set, param_name, min=None, max=None, value=None, vary=None, expr=None):
        """
        Auxilliary function to change a single parameter inside a Parameters() object.

        Note: changing one of the attributes to "None" does not work.

        :param param_set: A Parameters() object containing the parameter that will be changed
        :param param_name: The parameter that will be changed
        :param min:  The new min for the parameter. If not given, this attribute will not be changed.
        :param max: The new max for the parameter. If not given, this attribute will not be changed.
        :param value: The new value for the parameter. If not given, this attribute will not be changed.
        :param vary: The new vary state for the parameter. If not given, this attribute will not be changed.
        :param expr: The new expression string for the parameter. If not given, this attribute will not be changed.
        :return: None
        """

        if not min is None:
            param_set[param_name].min = min
        if not max is None:
            param_set[param_name].max = max
        if not value is None:
            param_set[param_name].value = value
        if not vary is None:
            param_set[param_name].vary = vary
        if not expr is None:
            param_set[param_name].expr = expr

    def calculate_band_norm(self, model):
        """
        Auxillary function used in the select_param_set() method. Calculates the difference between the data and the
        model, but only in the region around the resonances.

        :param model: A vector containing the values of the impedance of the model
        :return: cumulative difference (larger cum.diff. means worse fit)
        """
        #function to compare the two fit results
        freq = self.frequency_vector
        cumnorm = 0
        zone_factor = 1.2

        #check the bandwidth regions and check their least squares diff
        for it, band in enumerate(self.bandwidths):
            bandmask = np.logical_and((freq > band[0]/zone_factor), (freq < band[2]*zone_factor))
            raw_data  = abs(self.z21_data[bandmask])
            mdl1_data = abs(model[bandmask])
            norm1 = np.linalg.norm(raw_data - mdl1_data)
            cumnorm += norm1

        return cumnorm

    def plot_curve(self, param_set, order, main_res, title = None, angle = False):
        """
        Auxillary function to generate a plot of the given measurement data and the impedance of the model.

        :param param_set: A Parameters() object containing the circuits of the model
        :param order: The order of the model (i.e. the number of resonance circuits)
        :param main_res: Boolean. If True the function will only plot the main resonance
        :param title: Optional. String to use as title for the plot
        :return: None
        """

        testdata = self.calculate_Z(param_set, self.frequency_vector, 2, order, main_res, 2)
        if angle:
            plt.figure()
            plt.semilogx(self.frequency_vector, np.rad2deg(np.angle(self.z21_data)))
            plt.semilogx(self.frequency_vector, np.rad2deg(np.angle(testdata)))
            if title is not None:
                plt.title(title)
        else:
            plt.figure()
            plt.loglog(self.frequency_vector, abs(self.z21_data))
            plt.loglog(self.frequency_vector, abs(testdata))
            if title is not None:
                plt.title(title)

    def calc_Z_simple_RLC(self, parameters, freq, data, ser_par, mode):
        """
        Auxillary function that calculates the impedance of a single RLC circuit.

        This method is invoked by the model_bandwidth() funtion and it is an objective function like calculate_Z().

        :param parameters: A Parameters() object, containing R, L and C
        :param freq: The frequency vector over which the impedance is required
        :param data: The corresponding data; needed if FIT mode is selected.
        :param ser_par: Whether the RLC is a serial or parallel resonant circuit (1=serial;2=parallel)
        :param mode: Can be FIT or OUTPUT, selects what will be returned.
        :return: if mode == FIT, will return the difference in magnitude between the data and the model;
                if mode == OUTPUT, will return the impedance of the resonant circuit
        """
        w = np.pi*2*freq
        Z_R = parameters['R'].value
        Z_L = parameters['L'].value * 1j * w
        Z_C = 1 / (parameters['C'].value * 1j * w)

        match ser_par:
            case 1:#serial
                Z = Z_R + Z_L + Z_C
            case 2:#parallel
                Z = 1/(1/Z_R + 1/Z_C + 1/Z_L)

        match mode:
            case constants.fcnmode.FIT:
                # diff = (np.real(data) - np.real(Z)) + (np.imag(data) - np.imag(Z))
                # diff = np.linalg.norm(data-Z)
                diff = abs(data) - abs(Z)
                if constants.DEBUG_BW_MODEL_VERBOSE:
                    test_data = self.calc_Z_simple_RLC(parameters, freq, [], ser_par, 2)
                    plt.loglog(freq,abs(test_data))
                return (diff)
            case constants.fcnmode.OUTPUT:
                return Z

    def model_bandwidth(self, freqdata, data, peakfreq):
        """
        Auxillary function to brute-force fit a resonance circuit around a peak.

        :param freqdata: The frequency vector for said peak
        :param data: The measurement data around the peak
        :param peakfreq: the resonant frequency of the peak
        :return: Tuple: [b_l, b_u, R, L, C] The lower and upper 3dB points from the model (in Hz) and the R, L and C
            values for a circuit that models the band
        """


        #get the height of the peak and the index(will be used later)
        peakindex = np.argwhere(freqdata >= peakfreq)[0][0]
        peakheight = abs(data[peakindex])
        r_val = abs(peakheight)

        #set the flag for parallel/serial circuit
        match self.fit_type:
            case constants.El.INDUCTOR:
                ser_par_flag = 2
            case constants.El.CAPACITOR:
                ser_par_flag = 1

        #find pits in data
        match self.fit_type:
            case constants.El.INDUCTOR:
                pits = find_peaks(-abs(data),prominence=1e-3)
            case constants.El.CAPACITOR:
                pits = find_peaks(abs(data),prominence=1e-3)

        #get the indices of the pits closest to the peak (if they exist)
        try:
            lower_pit_index = pits[0][np.flipud(np.argwhere(pits[0]<peakindex))[0][0]]
        except:
            lower_pit_index = 0

        try:
            upper_pit_index = pits[0][(np.argwhere(pits[0]>peakindex))[0][0]]
        except:
            upper_pit_index = len(data)-1

        #crop data
        modelfreq = freqdata[lower_pit_index:upper_pit_index]
        modeldata = data[lower_pit_index:upper_pit_index]
        #rewrite the peak index after cropping
        peakindex = np.argwhere(modelfreq >= peakfreq)[0][0]

        try:
            #find the inflection points before and behind the peak in order to crop the peak
            # the derivatives need to be sav_gol filtered, otherwise they are too noisy
            ddt = np.gradient(abs(modeldata))
            ddt = scipy.signal.savgol_filter(ddt, 31,3)
            ddt2 = np.gradient(ddt)
            ddt2 = scipy.signal.savgol_filter(ddt2, 31, 3)
            signchange = np.argwhere(abs(np.diff(np.sign(ddt2))))

            #find left and right index of inflection points and crop the data (also constrain the data between 0 and max)
            left = np.clip((signchange[(signchange < peakindex)][-1] - 1), a_min = 0)
            right = np.clip((signchange[(signchange > peakindex)][0] + 1), a_max = len(modelfreq))
            modelfreq = modelfreq[left:right]
            modeldata = modeldata[left:right]
        except:
            #if the derivative approach does not work, we just do it the way it was before (this has less accuracy overall,
            # but should work for most cases)
            pass

        #space through the usual capacitance values in logarithmic steps
        C_max_exp = 3
        C_min_exp = -15
        numsteps = ((C_max_exp-C_min_exp) + 1) + 25
        C_values = np.flipud(np.logspace(C_min_exp, C_max_exp, num=numsteps))
        #C has to be set to some value, so the expr_string for L works
        C = C_values[-1]

        temp_params = Parameters()

        w_c2 = (peakfreq*2*np.pi)**2

        expr_string_L = '1/(C*'+ str(w_c2)+')'
        temp_params.add('R', value=abs(r_val), min = abs(r_val)*0.8,max=abs(r_val)*1.25,vary=False)
        # temp_params.add('L', value = L,  min = L*1e-3, max = L*1e3)
        temp_params.add('C',value = C)#, min = C*1e-3, max = C*1e6)
        temp_params.add('L',expr=expr_string_L)


        ################################################################################################################
        if constants.DEBUG_BW_MODEL_VERBOSE:
            plt.figure()
            plt.loglog(freqdata,abs(data))
            plt.ylim([min(abs(data))-0.5, max(abs(data))+0.5])
        ################################################################################################################


        # now step through the C values and look at the diff from the objective function in order to obtain a good
        # initial guess for lsq fitting

        diff_array = []
        for it,C_val in enumerate(C_values):
            temp_params.add('C',value = C_val, min = C_val*1e-3, max = C_val*1e6)
            diff_data = self.calc_Z_simple_RLC(temp_params, modelfreq, modeldata, ser_par_flag, 1)
            diff_array.append(sum((diff_data)))

        #check if we have a zero crossing
        if any(np.signbit(diff_array) == True):
            sign_change_index = np.argwhere(np.signbit(diff_array) == True)[0][0]
            #do an interpolation in order to find the cap value at the zero crossing -> most accurate value for C
            x = np.linspace(C_values[sign_change_index - 1], C_values[sign_change_index], 10000)
            y = np.linspace(diff_array[sign_change_index - 1], diff_array[sign_change_index], 10000)
            sign_change_index_interp = np.argwhere(np.signbit(y) == True)[0][0]
            C_val_rough_fit = x[sign_change_index_interp]

        else:
            #TODO: handle this case; should not be invoked, when cap values are stepped through well
            # maybe throw an exception and all that... and think about what BW to take then
            pass

        #TODO: look into what to do if the max and min values are too close to each other
        temp_params.add('C',value=C_val_rough_fit, min=C_val_rough_fit * 0.1, max=C_val_rough_fit * 10)

        #do a fit then after we have the approximate value of the cap
        out = minimize(self.calc_Z_simple_RLC, temp_params, args=(modelfreq,modeldata,ser_par_flag,1),
                            method='powell', options={'xtol': 1e-18, 'disp': True})

        ################################################################################################################
        # #PLOTS ( for when you are in the mood for visual analysis ¯\_(ツ)_/¯ )
        if constants.DEBUG_BW_MODEL:
            test_data_again = self.calc_Z_simple_RLC(out.params,freqdata,[],ser_par_flag,2)
            test_data_again_rough = self.calc_Z_simple_RLC(temp_params,freqdata,[],ser_par_flag,2)
            plt.figure()
            plt.plot(diff_array, marker = "D")
            plt.figure()
            plt.loglog(freqdata,abs(data))
            plt.loglog(freqdata,abs(test_data_again))
            plt.loglog(freqdata,abs(test_data_again_rough))

        ################################################################################################################

        #now get the bandwidth
        freq_interp = np.linspace(min(freqdata)-min(freqdata)*(1/1.5),max(freqdata)+max(freqdata)*1.5,num= 10000)
        match self.fit_type:
            case constants.El.INDUCTOR:
                data_interp = self.calc_Z_simple_RLC(out.params, freq_interp, [], ser_par_flag, constants.fcnmode.OUTPUT)
                BW_3_dB_height = peakheight * (1/np.sqrt(2))

                #get the 3dB-Points of the modeled curve
                b_u = freq_interp[np.argwhere(np.logical_and(abs(data_interp) < BW_3_dB_height, freq_interp > peakfreq))[0][0]]
                b_l = freq_interp[np.argwhere(np.logical_and(abs(data_interp) < BW_3_dB_height, freq_interp < peakfreq))[-1][0]]

                return [b_l, b_u, out.params['R'].value, out.params['L'].value, out.params['C'].value]




            case constants.El.CAPACITOR:
                data_interp = self.calc_Z_simple_RLC(out.params, freq_interp, [], ser_par_flag, constants.fcnmode.OUTPUT)

                BW_3_dB_height = peakheight * np.sqrt(2)

                # get the 3dB-Points of the modeled curve
                b_u = freq_interp[np.argwhere(np.logical_and(abs(data_interp) > BW_3_dB_height, freq_interp > peakfreq))[0][0]]
                b_l = freq_interp[np.argwhere(np.logical_and(abs(data_interp) > BW_3_dB_height, freq_interp < peakfreq))[-1][0]]

                return [b_l, b_u, out.params['R'].value, out.params['L'].value, out.params['C'].value]

    def fix_main_resonance_parameters(self, param_set):
        """
        Auxillary function to lock the main resonance parameters in place

        :param param_set: A Parameters() object containing the main resonance parameters that shell be fixed
        :return: None
        """
        param_set['R_s'].vary = False
        param_set['L'].vary = False
        param_set['C'].vary = False
        match self.fit_type:
            case constants.El.INDUCTOR:
                param_set['R_Fe'].vary = False
            case constants.El.CAPACITOR:
                param_set['R_iso'].vary = False

    def free_parameters_higher_order(self, param_set):
        """
        Auxillary function to free (i.e. set vary to True) the higher order circuit parameters.

        :param param_set: A Parameters object, containing the higher order resonances
        :return: None
        """

        for key_number in range(1, self.order + 1):
            #create keys
            C_key   = "C%s" % key_number
            L_key   = "L%s" % key_number
            R_key   = "R%s" % key_number
            w_key   = "w%s" % key_number
            param_set[C_key].vary = True
            param_set[L_key].vary = True
            param_set[R_key].vary = True
            param_set[w_key].vary = True

    def fix_parameters(self, param_set, R = True, L=True, C=True, w=True):
        """
        Auxilliary function to lock parameters in a Parameters() object

        :param param_set: A Parameters() object containing the parameters to lock
        :param R: Boolean. Whether to lock these parameters or not. True locks the parameters. Will lock all higher order Rs
        :param L: Boolean. Whether to lock these parameters or not. True locks the parameters. Will lock all higher order Ls
        :param C: Boolean. Whether to lock these parameters or not. True locks the parameters. Will lock all higher order Cs
        :param w: Boolean. Whether to lock these parameters or not. True locks the parameters. Will lock all higher order ws
        :return: None
        """

        param_set['R_s'].vary = False
        param_set['C'].vary = False
        param_set['L'].vary = False
        match self.fit_type:
            case constants.El.CAPACITOR:
                param_set['R_iso'].vary = False
            case constants.El.INDUCTOR:
                param_set['R_Fe'].vary = False
        for key_number in range(1, self.order + 1):
            #create keys
            C_key   = "C%s" % key_number
            L_key   = "L%s" % key_number
            R_key   = "R%s" % key_number
            w_key   = "w%s" % key_number
            param_set[C_key].vary = not C
            param_set[L_key].vary = not L
            param_set[R_key].vary = not R
            param_set[w_key].vary = not w


    ################################V EXPERIMENTAL V####################################################################

    def create_brutebank_model(self, param_set, number_circuits, initial_q):

        freq = self.frequency_vector
        data = self.z21_data

        banks = np.logspace(min(np.log10(freq)), max(np.log10(freq)), number_circuits + 2)[1:-1]

        indices = [np.argwhere(freq == [min(freq, key=lambda x: abs(x - bank)) for bank in banks][i])[0][0] for i in np.arange(len(banks))]

        datapoints = data[indices]
        frequencies = freq[indices]

        for it, center_freq in enumerate(frequencies):



                w_c = center_freq*2*np.pi
                #TODO: find a metric that represents how much the circuits interact with each other and set the resistance
                # values to appropriately high values EDIT: I'm going with factor 10 for now
                R = abs(datapoints[it]) * number_circuits/5
                C = 1 / (initial_q * R * w_c)

                BW = (w_c/initial_q)/2*np.pi
                if it != 0:
                    # create keys
                    C_key = "C%s" %   (it)
                    L_key = "L%s" %   (it)
                    R_key = "R%s" %   (it)
                    w_key = "w%s" %   (it)
                    BW_key = "BW%s" % (it)
                else:
                    C_key = "C"
                    L_key = "L"
                    R_key = "R_s"
                    w_key = "w"
                    BW_key = "BW"
                    param_set.add("R_iso", value=10e6, vary=False)

                expression_string_L = '(1/((' + C_key + '*' + str(config.CAPUNIT) + ')*(' + w_key + '*' + str(
                    config.FUNIT) + ')**2))/' + str(config.INDUNIT)

                C = C/config.CAPUNIT
                w_c = w_c/config.FUNIT
                BW= BW/config.FUNIT

                param_set.add(C_key, min=C*0.6, max=C*1.33, value=C, vary=True)
                param_set.add(R_key, value=R, min=R * 0.2, max=R * 5)
                param_set.add(w_key, min=w_c*0.8, max=w_c*1.2, value=w_c, vary=True)
                param_set.add(L_key, expr=expression_string_L)
                param_set.add(BW_key, min=BW * 0.8, max=BW * 1.2, value=BW, vary=False)

        self.order = number_circuits -1
        return param_set

    def fit_brutebank_model(self, param_set):
        freq = self.frequency_vector
        data = self.z21_data
        fit_order = self.order
        mode = constants.fcnmode.FIT_LOG

        modelfreq = freq[freq<=config.FREQ_UPPER_LIMIT]
        modeldata = data[freq<=config.FREQ_UPPER_LIMIT]

        fit_main_resonance = 0
        out = minimize(self.calculate_Z, param_set,
                        args=(modelfreq, modeldata, fit_order, fit_main_resonance, mode,),
                        method='powell', options={'xtol': 1e-18, 'disp': True})

        return out.params





