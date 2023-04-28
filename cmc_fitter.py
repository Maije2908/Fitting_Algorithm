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
        self.cmcmodel = None

    def set_file(self, file_dict):
        self.file_dict = file_dict

    def calc_impedances(self, Z0):
        for key in self.file_dict:
            super().set_file(self.file_dict[key])
            super().calc_series_thru(Z0)
            self.data_dict[key] = self.z21_data

    def smooth_data(self):
        for key in self.file_dict:
            self.z21_data = self.data_dict[key]
            super().smooth_data()
            self.smooth_data_dict[key] = [self.data_mag, self.data_ang]

    def one_sided_params_to_sym_params(self):
        '''
        function to convert the one sided parameters obtained from the fit to two sided parameters that can be used in
        a Spice type circuit simulator
        :return:
        '''
        #TODO: this method is still in progress
        symparams = {}
        for key in self.params_dict:
            two_sided = {}
            if key == 'DM':
                for param_key in self.params_dict[key]:
                    #calculate scaling factors for the parameters (so that they are in the correct unit)
                    #TODO: consider the different cases for elements in the parameter set (e.g. L_p vs L)
                    if 'L' in param_key:
                        scaling = config.INDUNIT
                    elif 'C' in param_key:
                        scaling = config.CAPUNIT
                    elif 'w' in param_key:
                        scaling = config.FUNIT

                    two_sided[param_key] = self.params_dict[key][param_key]

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
        for key in [k for k in self.file_dict.keys()]:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            super().get_resonances()
            self.order_dict[key] = self.order
            self.bandwidth_dict[key] = [self.bandwidths, self.bad_bandwidth_flag, self.peak_heights]

    def create_nominal_parameters_CM(self):

        key = 'CM'
        params = Parameters()

        self.data_mag = self.smooth_data_dict[key][0]
        self.data_ang = self.smooth_data_dict[key][1]
        self.z21_data = self.data_dict[key]
        self.f0 = self.main_res_dict[key][0]
        self.f0_index = self.main_res_dict[key][1]

        self.params_dict[key] = Parameters()
        self.nominal_value = self.nominals_dict[key]

        match self.cmcmodel:
            case cmctype.NANOCRYSTALLINE:
                #TODO!!!!!
                pass
            case cmctype.MULTIRESONANCE:
                #TODO!!!!!
                pass
            case cmctype.PLATEAU:
                params.add('L', value = self.nominals_dict['CM'])
                params.add('R', value = abs(self.data_dict['CM'][self.main_res_dict['CM'][1]]))
                pass

        self.params_dict[key] = super().create_nominal_parameters(self.params_dict[key])
        #clear some parameters we obtain from the super() method
        # self.params_dict[key].pop('R_s') EDIT: we might need R_s
        self.params_dict[key]['L'].expr = ''
        self.params_dict[key]['L'].vary = True
        self.params_dict[key]['R_s'].vary = False
        #set new boundaries for our parameters
        self.params_dict[key]['C'].max = self.params_dict[key]['C'].value * 1e2
        self.params_dict[key]['C'].min = self.params_dict[key]['C'].value * 1e-6
        self.params_dict[key]['L'].max = self.params_dict[key]['L'].value * 1.01e0
        self.params_dict[key]['L'].min = self.params_dict[key]['L'].value * 0.98e0
        self.params_dict[key]['R_Fe'].max = self.params_dict[key]['R_Fe'].value * 1e3
        self.params_dict[key]['R_Fe'].min = self.params_dict[key]['R_Fe'].value * 1e-3


    def create_nominal_parameters_DM(self, param_set = None):
        for key in [k for k in self.file_dict.keys()]:

            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]
            self.fit_type = El.INDUCTOR

            self.params_dict[key] = Parameters()
            self.nominal_value = self.nominals_dict[key]

            self.params_dict[key] = super().create_nominal_parameters(self.params_dict[key])
            #clear some parameters we obtain from the super() method
            # self.params_dict[key].pop('R_s') EDIT: we might need R_s
            self.params_dict[key]['L'].expr = ''
            self.params_dict[key]['L'].vary = True
            self.params_dict[key]['R_s'].vary = False
            #set new boundaries for our parameters
            self.params_dict[key]['C'].max = self.params_dict[key]['C'].value * 1e2
            self.params_dict[key]['C'].min = self.params_dict[key]['C'].value * 1e-6
            self.params_dict[key]['L'].max = self.params_dict[key]['L'].value * 1.01e0
            self.params_dict[key]['L'].min = self.params_dict[key]['L'].value * 0.98e0
            self.params_dict[key]['R_Fe'].max = self.params_dict[key]['R_Fe'].value * 1e3
            self.params_dict[key]['R_Fe'].min = self.params_dict[key]['R_Fe'].value * 1e-3


            if key == "DM":
                self.params_dict[key].add('C_p', value=7.95e-12 / config.CAPUNIT, min=1e-13 / config.CAPUNIT, max=1e-8 / config.CAPUNIT, vary=True)
                # #TODO: this is hardcoded
                #leakage inductor
                # L_main = self.params_dict[key]['L'].value #NOTE: L is already in INDUNIT; DON'T RESCALE!!!!
                # self.params_dict[key].add('L_leak', value=L_main/10, vary=True, min=L_main/1e3, max=L_main)

                #TODO: check if OC exists

                C_p = self.nominals_dict['OC']/2
                wc = self.main_res_dict['OC'][0] * 2 * np.pi
                L_p = (1/(wc**2*C_p))*2
                R_p = 2*abs(self.data_dict['OC'][self.main_res_dict['OC'][1]])
                self.params_dict[key].add('C_p', value=C_p / config.CAPUNIT, min=C_p*1e-1 / config.CAPUNIT, max=C_p*1e1 / config.CAPUNIT, vary=True)
                # self.params_dict[key].add('L_p', value=L_p / config.INDUNIT, min=1e-13 / config.INDUNIT, max=1e-6 / config.INDUNIT, vary=True)
                self.params_dict[key].add('R_p', value=R_p, min = 1e-6, vary=False)


                # self.params_dict[key].add('C_p', value=C_p / config.CAPUNIT, min=C_p*1e-2 / config.CAPUNIT,max=C_p*1e2 / config.CAPUNIT, vary=True)
                # self.params_dict[key].add('L_p', value = L_p/config.INDUNIT, min=L_p*1e-2 / config.INDUNIT, max=L_p*1e2 / config.INDUNIT, vary=True)
                # self.params_dict[key].add('R_p', value=R_p, vary=True, min=R_p*1e-3, max=R_p*1e3)

                # #parallel capacitances model for vogt j45
                # #150;1.33e-6;7.95e-12;
                # self.params_dict[key].add('R_p', value = 150, vary = True, min = 50, max =1500)
                # self.params_dict[key].add('L_p', value = 1.4e-6/config.INDUNIT, min=1e-7 / config.INDUNIT, max=1e-5 / config.INDUNIT, vary=True)
                # self.params_dict[key].add('C_p', value = 7.95e-12/config.CAPUNIT, min=1e-13 / config.CAPUNIT, max=1e-10 / config.CAPUNIT, vary=True)

    def fit_cmc_main_res(self):
        for key in [k for k in self.file_dict.keys() if k != 'OC']:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.nominal_value = self.nominals_dict[key]

            freq = self.frequency_vector
            data = self.data_dict[key]

            #frequency limit data
            #TODO: this is HARDCODED for the vogt j45 test
            mask = [freq<2e8][0]
            freq = freq[mask]
            data = data[mask]

            fit_order = 0
            fit_main_resonance = 1
            mode = FIT_BY

            out = minimize(self.calculate_Z_CMC, self.params_dict[key],
                           args=(freq, data, fit_order, fit_main_resonance, mode, key),
                           method='powell', options={'xtol': 1e-18, 'disp': True})
            # out = minimize(self.calculate_Z_CMC, self.params_dict[key],
            #                args=(freq, data, fit_order, fit_main_resonance, mode, key),
            #                method='dual_annealing')#, options={'xtol': 1e-18, 'disp': True})
            self.params_dict[key] = out.params
            pass

            # self.params_dict[key] = super().fit_main_res_inductor_file_1(self.params_dict[key])

    def create_higher_order_parameters(self):
        for key in [k for k in self.file_dict.keys() if k != 'OC']:
            # if key == "DM":
            #     self.bandwidth_dict["DM"][0].pop(0)
            #     self.bandwidth_dict["DM"][1] = self.bandwidth_dict["DM"][1][1:]
            #     self.bandwidth_dict["DM"][2].pop(0)
            #     self.order_dict["DM"] -= 1

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
        for key in [k for k in self.file_dict.keys() if k != 'OC']:
            self.data_mag = self.smooth_data_dict[key][0]
            self.data_ang = self.smooth_data_dict[key][1]
            self.z21_data = self.data_dict[key]
            self.f0 = self.main_res_dict[key][0]
            self.f0_index = self.main_res_dict[key][1]

            self.nominal_value = self.nominals_dict[key]
            self.order = self.order_dict[key]
            self.modeled_bandwidths = self.bandwidth_dict[key][0]

            self.params_dict[key] = super().correct_parameters(self.params_dict[key], change_main=0, num_it=4)
            self.params_dict[key] = self.fix_main_resonance_parameters(self.params_dict[key])

            # self.fit_cmc_main_res()
            self.params_dict[key] = super().pre_fit_bands(self.params_dict[key])
            self.params_dict[key] = super().fit_curve_higher_order(self.params_dict[key])

    def fix_main_resonance_parameters(self, param_set):
        param_set['C'].vary = False
        param_set['L'].vary = False
        param_set['R_Fe'].vary = False
        return param_set

    def plot_curves_cmc(self, mainres):
        for key in self.file_dict:

            freq = self.frequency_vector
            data = self.data_dict[key]

            if key == 'OC':
                params = self.params_dict['DM']
                order = self.order_dict['DM']
            else:
                params = self.params_dict[key]
                order = self.order_dict[key]



            model_data = self.calculate_Z_CMC(params, self.frequency_vector, 0, order,
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
        Zmain = 0
        Z=0

        Zmain = 1 / (1/R_Fe + 1/XL + 1/XC)


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
            Zmain += 1 / ((1 / Z_C) + (1 / Z_L) + (1 / Z_R))

        if meas_type == 'CM':
            Zmain += parameters['R_s'].value
            Z = Zmain

        if meas_type == 'DM' or meas_type == 'OC':

            C_par = parameters['C_p'].value * config.CAPUNIT
            # L_par = parameters['L_p'].value * config.INDUNIT
            R_par = parameters['R_p'].value

            # L_leak = parameters['L_leak'].value

            # #configuration with R_s
            # Rs =  parameters['R_s'].value
            # Z_par = 1/(1j*w*C_par)
            # Z_par = 1/(1j*w*C_par)+R_par+1j*w*L_par
            # Z_terminal = (2*Rs*Z_par)/(2*Rs + Z_par)
            # Zmain += Z_terminal
            # Z_main = (Z_par*Zmain)/(Z_par+Zmain)
            # Z = 2*Rs + Zmain

            #configuration without R_s
            Z_par =  1/(1j*w*C_par) #+ 1j*w*L_par + R_par

            # Z_leak = 1j*w*L_leak
            # Zmain += Z_leak


            if meas_type == 'DM':
                Z = Zmain #1/(1/Zmain + 1/Z_par)

            if meas_type == 'OC':
                Z = 1/((1/Z_par) + (1/(Zmain+Z_par)))




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

    def calc_parallel_C(self):
        #TODO: zombie function as of yet
        data = self.smooth_data_dict["OC"][0]
        freq = self.frequency_vector







