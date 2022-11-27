

from fitter import *
import config
from lmfit import Parameters
import constants

#derive a class from the fitter
class CMC_Fitter(Fitter):
    def __init__(self, logger):
        #init the Fitter class
        super().__init__(logger)
        #specific variables for CMC
        self.data_dict = {}
        self.file_dict = None
        self.fit_type = constants.El.INDUCTOR
        self.smooth_data_dict = {}
        self.nominals_dict = {}
        self.main_res_dict = {}
        self.bandwidth_dict = {}
        self.order_dict = {}
        self.params_dict = {}
        #TODO: Series resistance is hardcoded to one Ohm
        self.series_resistance = 1


    def set_file(self, file_dict):
        self.file_dict = file_dict

    def calc_series_thru(self, Z0):
        for key in self.file_dict:
            super().set_file(self.file_dict[key])
            super().calc_series_thru(Z0)
            self.data_dict[key] = self.z21_data

    def smooth_data(self):
        for key in self.file_dict:
            self.z21_data = self.data_dict[key]
            super().smooth_data()
            self.smooth_data_dict[key] = [self.data_mag, self.data_ang]
        pass

    def calculate_nominal_value(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            super().calculate_nominal_value()
            self.nominals_dict[key] = self.nominal_value

    def get_main_resonance(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            super().get_main_resonance()
            self.main_res_dict[key] = [self.f0, self.f0_index]

    def get_resonances(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            super().get_resonances()
            self.order_dict[key] = self.order
            self.bandwidth_dict[key] = [self.bandwidths, self.bad_bandwidth_flag, self.peak_heights]

    def create_nominal_parameters(self, param_set = None):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.params_dict[key] = Parameters()
            self.nominal_value = self.nominals_dict[key]

            self.params_dict[key] = super().create_nominal_parameters(self.params_dict[key])

    def fit_cmc_main_res(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.nominal_value = self.nominals_dict[key]

            self.params_dict[key] = super().fit_main_res_inductor_file_1(self.params_dict[key])

    def create_higher_order_parameters(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]
            self.order = self.order_dict[key]
            self.bandwidths = self.bandwidth_dict[key][0]
            self.bad_bandwidth_flag = self.bandwidth_dict[key][1]
            self.peak_heights = self.bandwidth_dict[key][2]

            self.params_dict[key] = super().create_higher_order_parameters(2,self.params_dict[key])
            self.bandwidth_dict[key][0] = self.modeled_bandwidths

    def fit_cmc_higher_order_res(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.nominal_value = self.nominals_dict[key]
            self.order = self.order_dict[key]
            self.modeled_bandwidths = self.bandwidth_dict[key][0]

            self.params_dict[key] = super().correct_parameters(self.params_dict[key], 0, 2)
            self.params_dict[key] = super().pre_fit_bands(self.params_dict[key])
            self.params_dict[key] = super().fit_curve_higher_order(self.params_dict[key])

    def plot_curves_cmc(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            super().plot_curve(self.params_dict[key], self.order_dict[key], 0, key)







