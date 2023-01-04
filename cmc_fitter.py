import matplotlib.pyplot as plt

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
            #clear some parameters we obtain from the super() method
            # self.params_dict[key].pop('R_s') EDIT: we might need R_s
            self.params_dict[key]['L'].expr = ''
            self.params_dict[key]['L'].vary = True
            #set new boundaries for our parameters
            self.params_dict[key]['C'].max = self.params_dict[key]['C'].value * 1e3
            self.params_dict[key]['C'].min = self.params_dict[key]['C'].value * 1e-3
            self.params_dict[key]['L'].max = self.params_dict[key]['L'].value * 1e3
            self.params_dict[key]['L'].min = self.params_dict[key]['L'].value * 1e-3
            self.params_dict[key]['R_Fe'].max = self.params_dict[key]['R_Fe'].value * 1e3
            self.params_dict[key]['R_Fe'].min = self.params_dict[key]['R_Fe'].value * 1e-3


            if key == "DM":
                #TODO: this is hardcoded
                self.params_dict[key].add('R_p', value = 150, vary = False)
                self.params_dict[key].add('L_p', value = 1.33e-6/config.INDUNIT, vary = False)
                self.params_dict[key].add('C_p', value = 7.95e-12/config.CAPUNIT, vary = False)

    def fit_cmc_main_res(self):
        for key in self.file_dict:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.nominal_value = self.nominals_dict[key]

            freq = self.frequency_vector
            data = self.data_dict[key]
            fit_order = 0
            fit_main_resonance = 1
            mode = FIT_BY

            out = minimize(self.calculate_Z_CMC, self.params_dict[key],
                           args=(freq, data, fit_order, fit_main_resonance, mode, key),
                           method='powell', options={'xtol': 1e-18, 'disp': True})
            self.params_dict[key] = out.params
            pass

            # self.params_dict[key] = super().fit_main_res_inductor_file_1(self.params_dict[key])

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

    def plot_curves_cmc(self, mainres):
        for key in self.file_dict:

            freq = self.frequency_vector
            data = self.data_dict[key]

            model_data = self.calculate_Z_CMC(self.params_dict[key], self.frequency_vector, 0, self.order_dict[key],
                                              mainres, constants.fcnmode.OUTPUT, key )
            plt.figure()
            plt.loglog(freq,abs(data))
            plt.loglog(freq,abs(model_data))
            plt.title(key)

    def calculate_Z_CMC(self, parameters, frequency_vector, data, fit_order, fit_main_res, modeflag, meas_type):
        # if we only want to fit the main resonant circuit, set order to zero to avoid "for" loops
        if fit_main_res:
            order = 0
        else:
            order = fit_order

        # create array for frequency
        freq = frequency_vector
        w = freq * 2 * np.pi

        # get parameters for main circuit
        C = parameters['C'].value * config.CAPUNIT
        L = parameters['L'].value * config.INDUNIT
        R_Fe = parameters['R_Fe'].value


        # calculate main circuits resistance
        XC = 1 / (1j * w * C)
        XL = 1j * w * L
        Z = 0

        Z = 1 / (1/R_Fe + 1/XL + 1/XC)


        for actual in range(1, order + 1):
            key_number = actual
            C_key = "C%s" % key_number
            L_key = "L%s" % key_number
            R_key = "R%s" % key_number
            C_act = parameters[C_key].value * config.CAPUNIT
            L_act = parameters[L_key].value * config.INDUNIT
            R_act = parameters[R_key].value
            Z_C = 1 / (1j * w * C_act)
            Z_L = (1j * w * L_act)
            Z_R = R_act
            Z += 1 / ((1 / Z_C) + (1 / Z_L) + (1 / Z_R))

        if meas_type == 'DM':
            C_par = parameters['C_p'].value * config.CAPUNIT
            R_par = parameters['R_p'].value
            L_par = parameters['L_p'].value * config.INDUNIT

            Z_par = R_par + 1j*w*L_par + 1/(1j*w*C_par)

            Z = 1/(1/Z + 1/Z_par)



        match modeflag:
            case fcnmode.FIT:
                diff = abs(data) - abs(Z)
                return (diff)
            case fcnmode.FIT_LOG:
                diff = np.log10(abs(data)) - np.log10(abs(Z))
                return (diff)
            case fcnmode.OUTPUT:
                return Z
            case fcnmode.ANGLE:
                return np.angle(data) - np.angle(Z)
            case fcnmode.FIT_REAL:
                return np.real(data) - np.real(Z)
            case fcnmode.FIT_IMAG:
                return np.imag(data) - np.imag(Z)





