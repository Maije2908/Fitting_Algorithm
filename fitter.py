
# The fitter class shall take the task of fitting the data, as well as smoothing it and performing manipulations
# This is likely to become a rather long task, especially for CMCs and this class is therefore likely to be long
# I do not know yet what it will have to contain and how to best handle the data
# Most of this class will be based on Payer's program
#NOTE: THE CLASS IN THE FORM THAT IT IS NOW IS NOT ABLE TO MANAGE MULTIPLE FILES!!!!!




import numpy as np
import scipy
import matplotlib.pyplot as plt



class Fitter:

    def __init__(self):
        self.nominal_value = None
        self.parasitive_resistance = None
        self.prominence = None
        self.saturation = None
        self.files = None
        self.z21_series_thru = None
        self.data_mag = None
        self.data_ang = None

    ####################################################################################################################
    # Parsing Methods
    ####################################################################################################################

    #method to set the entry values of the specification
    def set_specification(self, pass_val, para_r, prom, sat):
        self.nominal_value = pass_val
        self.parasitive_resistance = para_r
        self.prominence = prom
        self.saturation = sat

    #method to parse the files from the iohandler
    def set_files(self, files):
        self.files = files



    ####################################################################################################################
    # Pre-Processing Methods
    ####################################################################################################################

    def calc_series_thru(self, Z0):
        for file in self.files:
            self.z21_series_thru = 2 * Z0 * ((1 - file.data.s[:,1,0]) / file.data.s[:,1,0])


    def smooth_data(self):
        # Use Savitzky-Golay filter for smoothing the input data, because in the region of the global minimum there is
        # oscillation. After filtering a global minimum can be found easier.
        sav_gol_mode = 'interp'
        self.data_mag = scipy.signal.savgol_filter(abs(self.z21_series_thru), 51, 2, mode=sav_gol_mode)
        self.data_ang = scipy.signal.savgol_filter(np.angle(self.z21_series_thru, deg=True), 51, 2, mode=sav_gol_mode)
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
                # TODO: V solve this via boolean indexing V
                for inductive_range in range(offset, len(freq)):
                    if self.data_ang[inductive_range] < 0:
                        break  # range until first sign change
                if max(self.data_ang[offset:inductive_range]) < 88:
                    raise Exception("Error: Inductive range not detected (max phase = {value}°).\n"
                                    "Please specify nominal inductance.".format(value=np.round(max(self.data_ang), 1)))
                for sample in range(offset, len(freq)):
                    if self.data_ang[sample] == max(self.data_ang[offset:inductive_range]):
                        self.nominal_value = self.data_mag[sample] / 2 / np.pi / freq[sample]
                        break
            case 2: #CAPACITOR
                for capacitive_range in range(offset, len(freq)):
                    if self.data_ang[capacitive_range] > 0:
                        break  # range until first sign change
                if min(self.data_ang[offset:capacitive_range]) > -88:
                    raise Exception("Error: Capacitive range not detected (min phase = {value}°).\n"
                                    "Please specify nominal capacitance.".format(value=np.round(min(self.data_ang), 1)))

                test_values = []
                for sample in range(offset, len(freq)):
                    if self.data_ang[sample] == min(self.data_ang[offset:capacitive_range]):
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