import config
import skrf as rf
import pandas as pd
import os
from fitter import *
import constants
import matplotlib
from matplotlib import pyplot as plt

class IOhandler:
    """
    The IOHandler class takes care of the filehandling.
    It loads files and stores a reference to it.
    It also takes care of the output, generating the LTSpice Netlists as well as generating the Bode-plots for output
    """

    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.files = list()
        self.outpath = None

    def set_out_path(self, path):
        """
        Setter method to set the output path for the IOhandler

        :param path: The requested output path
        :return: None
        """
        self.outpath = path

    def load_file(self, path):
        """
        Method to load a sNpfile (Touchstone file) from a given path.
        Loads the file's contents and stores it in an SNpFile class.

        :param path: The path of the file to be loaded
        :return: None
        :raises Exception: if loading the file did not work
        """

        try:
            for actual_path in path:
                ntwk = rf.Network(actual_path)

                #check if file is already loaded -> if so, skip it
                if ntwk.name in [file.name for file in self.files]:
                    self.logger.warning("Warning; file: \"" + ntwk.name + "\" already present, did not load!")
                    continue

                self.logger.info("Opened file: \"" + ntwk.name+"\"")
                self.files.append(ntwk)
        except Exception as e:
            raise e

    def generate_Netlist_2_port(self, parameters, fit_order, fit_type, saturation_table, captype = None):
        """
        Writes an LTSpice Netlist to the path that is stored in the IOhandlers instance variable.

        Will output an LTSpice Netlist for current dependent inductor/capacitor with
        **constant higher order resonances**. That is the higher order resonant circuits **will not** be current/voltage
        dependent in this form of output.


        :param parameters: The Parameters for the model. A Parameters() object containing the model parameters for
            reference file
        :param fit_order: The order of the model i.e. the number of circuits
        :param fit_type: Whether the element is a coil or capacitor
        :param saturation_table: The saturation table for the elements. A dict type object with a key equal to that of
            the parameter in question containing the saturation table as a string
        :param captype: The type of capacitor. Can be GENERIC or MLCC
        :return: None
        """

        out = parameters
        order = fit_order

        match fit_type:
            case constants.El.INDUCTOR:

                # define the name of the model here:
                model_name = "L_1"

                # main element parameters
                L = out['L'].value*config.INDUNIT
                C = out['C'].value*config.CAPUNIT
                R_s = out['R_s'].value
                R_p = out['R_Fe'].value

                lib = '* Netlist for Inductor Model {name} (L={value}H)\n' \
                      '* Including {number} Serially Chained Parallel Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                     value=str(L),
                                                                                                     number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'


                ############### HIGHER ORDER ELEMENTS ##################################################################
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value*config.CAPUNIT
                    Lx = out['L%s' % circuit].value*config.INDUNIT
                    Rx = out['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'
                    lib += 'C{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Cx) + "\n"
                    lib += 'L{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Lx) + "\n"
                    lib += 'R{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Rx) + "\n"

                ################ MAIN ELEMENT ##########################################################################
                # node connections from current dependent inductor model

                main_res_terminal_port = '1' if order > 0 else 'PORT2'
                lib += 'R_s PORT1 B1 ' + str(R_s) + "\n"
                lib += 'R_p B1 '+main_res_terminal_port+' R = limit({lo}, {hi}, {R_Fe} * V(K_Fe))'.format(lo = R_p * 1e-8, hi = R_p * 1e8, R_Fe = R_p) + "\n"
                lib += 'C PORT1 '+main_res_terminal_port+' ' + str(C) + "\n"
                lib += 'BL B1 '+main_res_terminal_port+' V=V(K_L)*V(L)' + "\n"

                #'test' inductor
                lib += 'L L 0 ' + str(L) + "\n"
                lib += 'F1 0 L B1 1' + "\n"

                ############## PROPORTIONALITY TABLES ##################################################################
                lib += '* The values for the Current-Inductance-Table can be edited here:' + "\n"
                #proportionality factor for L
                lib += '* current dependent proportionality factor for L' + "\n"
                lib += 'B2 K_L 0 V=table(abs(I(B1)),{table})'.format(table=saturation_table['L']) + "\n"
                #proportionality factor for R_Fe
                lib += '* current dependent proportionality factor for R_Fe' + "\n"
                lib += 'B3 K_FE 0 V=table(abs(I(B1)),{table})'.format(table=saturation_table['R_Fe']) + "\n"

                lib += '.ENDS {inductor}'.format(inductor=model_name) + "\n"


            case constants.El.CAPACITOR:

                # define the name of the model here:
                model_name = "C_1"

                # main element parameters
                C = out['C'].value*config.CAPUNIT
                Ls = out['L'].value*config.INDUNIT
                R_s = out['R_s'].value
                R_iso = out['R_iso'].value

                lib = '* Netlist for Capacitor Model {name} (C={value}F)\n' \
                      '* Including {number} Parallely Chained Serial Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                    value=str(C),
                                                                                                    number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                ############### MAIN ELEMENT ###########################################################################
                lib += 'R_s PORT1 LsRs R = limit({lo}, {hi}, {R_s} * V(K_Rs))'.format(lo = R_s * 1e-8,
                                                                                      hi = R_s * 1e8, R_s = R_s) + "\n"
                lib += 'R_iso Vcap 0 ' + str(R_iso) + "\n"
                lib += 'L_s LsRs Vcap ' + str(Ls) + "\n"
                lib += 'E1 E1 0 Vcap 0 1 ' + "\n"
                lib += 'C E1 0 ' + str(C) + "\n"
                lib += 'B1 0 Vcap I=I(E1)*V(K) ' + "\n"

                ############### ACOUSTIC RESONANCE PARAMETERS FOR MLCCs ################################################

                if captype == constants.captype.MLCC:
                    RA = out['R_A'].value
                    LA = out['L_A'].value*config.INDUNIT
                    CA = out['C_A'].value*config.CAPUNIT
                    # current dependent coil for higher order res:
                    lib += 'BL{no} PORT1 NL{node1} '.format(no='A', node1='A') + 'V=V(VL{no})*V(K_L{no})'.format(no='A') + "\n"
                    lib += 'L{no} VL{no} 0 '.format(no='A') + str(LA) + "\n"
                    lib += 'FL{no} 0 VL{no} BL{no} 1'.format(no='A') + "\n"

                    lib += '* current dependent proportionality factor for L{no}'.format(no='A') + "\n"
                    lib += 'BLK{no} K_L{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A',table=saturation_table['L_A']) + "\n"
                    lib += "\n"

                    # current dependent cap for higher order res:
                    lib += 'BC{no} NL{node1} NC{node1} '.format(no='A', node1='A') + 'I=-I(BCT{no})*V(K_C{no})'.format(no='A') + "\n"
                    lib += 'C{no} VC{no} 0 '.format(no='A') + str(CA) + "\n"
                    lib += 'BCT{no} VC{no} 0 '.format(no='A') + 'V=V(NL{node1})-V(NC{node1})'.format(node1='A') + "\n"

                    lib += '* current dependent proportionality factor for C{no}'.format(no='A') + "\n"
                    lib += 'BCK{no} K_C{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A',table=saturation_table['C_A']) + "\n"

                    # current dependent resistor
                    lib += 'R_{no} NC{node1} PORT2 R = limit({lo}, {hi}, {R_x} * V(K_R{no}))'.format(no='A',
                                                                                                     node1='A',
                                                                                                     lo=RA * 1e-8,
                                                                                                     hi=RA * 1e8,
                                                                                                     R_x=RA) + "\n"

                    lib += '* current dependent proportionality factor for R{no}'.format(no='A') + "\n"
                    lib += 'BRK{no} K_R{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A',table=saturation_table['R_A']) + "\n"
                    lib += "\n"

                ############### HIGHER ORDER ELEMENTS ##################################################################
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value*config.CAPUNIT
                    Lx = out['L%s' % circuit].value*config.INDUNIT
                    Rx = out['R%s' % circuit].value

                    lib += 'R{no} PORT1 {node2} '.format(no=circuit, node2=circuit) + str(Rx) + "\n"
                    lib += 'L{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=order + circuit) + str(
                        Lx) + "\n"
                    lib += 'C{no} {node1} 0 '.format(no=circuit, node1=order + circuit) + str(Cx) + "\n"

                ############## PROPORTIONALITY TABLES ##################################################################
                # proportionality factor for C
                lib += '* The values for the Voltage-Capacitance-Table can be edited here:' + "\n"
                lib += '* current dependent proportionality factor for L' + "\n"
                lib += 'B2 K 0 V=table(abs(V(PORT1)),{table}) '.format(table=saturation_table['C']) + "\n"
                # proportionality factor for R_Fe
                lib += '* current dependent proportionality factor for R_s' + "\n"
                lib += 'B3 K_Rs 0 V=table(abs(I(B1)),{table})'.format(table=saturation_table['R_s']) + "\n"

                lib += '.ENDS {name}'.format(name=model_name) + "\n"


        ############### OUTPUT #########################################################################################
        # get output folder and path
        out_path = os.path.split(self.outpath)[0]
        dir_name = os.path.normpath(self.outpath).split(os.sep)[-2]
        out_folder = os.path.join(out_path, "fit_results_%s" % dir_name)

        #create the folder; should not be necessary to handle an exception; however folder could be write protected
        try:
            os.makedirs(out_folder, exist_ok = True)
        except Exception:
            raise

        # write LTSpice .lib file
        file = open(os.path.join(out_folder, "LT_Spice_Model_" + dir_name + ".lib"), "w+")
        file.write(lib)
        file.close()

    def generate_Netlist_2_port_full_fit(self, parameters, fit_order, fit_type, saturation_table, captype=None):
        """
        Writes an LTSpice Netlist to the path that is stored in the IOhandlers instance variable.

        This method **does output fully parametric models**, i.e. the higher order resonant circuits **will be**
        current/voltage dependent as well as the main element.

        :param parameters: The Parameters for the model. A Parameters() object containing the parameters of the
            reference file
        :param fit_order: The order of the model i.e. the number of circuits
        :param fit_type: Whether the element is a coil or capacitor
        :param saturation_table: The saturation table for the elements. A dict type object with a key equal to that of
            the parameter in question containing the saturation table as a string
        :param captype: The type of capacitor. Can be GENERIC or MLCC
        :return: None
        """

        out = parameters
        order = fit_order

        match fit_type:
            case constants.El.INDUCTOR:

                # define the name of the model here:
                model_name = "L_1"

                # parameters for the main elements
                L = out['L'].value*config.INDUNIT
                C = out['C'].value*config.CAPUNIT
                R_s = out['R_s'].value
                R_p = out['R_Fe'].value

                lib = '* Netlist for Inductor Model {name} (L={value}H)\n' \
                      '* Including {number} Serially Chained Parallel Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                     value=str(L), number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                ############### HIGHER ORDER ELEMENTS ##################################################################
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value*config.CAPUNIT
                    Lx = out['L%s' % circuit].value*config.INDUNIT
                    Rx = out['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'

                    #current dependent coil for higher order res:
                    lib += 'BL{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + 'V=V(VL{no})*V(K_L{no})'.format(no=circuit) + "\n"
                    lib += 'L{no} VL{no} 0 '.format(no=circuit) + str(Lx) + "\n"
                    lib += 'FL{no} 0 VL{no} BL{no} 1'.format(no=circuit) + "\n"
                    #"test" inductor
                    lib += '* current dependent proportionality factor for L{no}'.format(no=circuit) + "\n"
                    lib += 'BLK{no} K_L{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['L%s' % circuit]) + "\n"
                    lib += "\n"

                    # current dependent cap for higher order res:
                    lib += 'BC{no} {node1} {node2} '.format(no=circuit, node1=circuit,node2=node2) + 'I=-I(BCT{no})*V(K_C{no})'.format(no=circuit) + "\n"
                    lib += 'C{no} VC{no} 0 '.format(no=circuit) + str(Cx) + "\n"
                    lib += 'BCT{no} VC{no} 0 '.format(no=circuit) + 'V=V({node1})-V({node2})'.format(node1=circuit, node2=node2) + "\n"
                    #"test" cap
                    lib += '* current dependent proportionality factor for C{no}'.format(no=circuit) + "\n"
                    lib += 'BCK{no} K_C{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['C%s' % circuit]) + "\n"
                    lib += "\n"

                    #current dependent resistor
                    lib += 'R_{no} {node1} {node2} R = limit({lo}, {hi}, {R_x} * V(K_R{no}))'.format(no=circuit, node1=circuit,node2=node2, lo=Rx * 1e-12, hi=Rx * 1e8,R_x=Rx) + "\n"

                    lib += '* current dependent proportionality factor for R{no}'.format(no=circuit) + "\n"
                    lib += 'BRK{no} K_R{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['R%s' % circuit]) + "\n"
                    lib += "\n"

                ############### MAIN ELEMENT ###########################################################################
                main_res_terminal_port = '1' if order > 0 else 'PORT2'
                lib += 'R_s PORT1 B1 ' + str(R_s) + "\n"
                lib += 'R_p B1 '+main_res_terminal_port+' R = limit({lo}, {hi}, {R_Fe} * V(K_Fe))'.format(lo = R_p * 1e-8, hi = R_p * 1e8, R_Fe = R_p) + "\n"
                lib += 'C PORT1 '+main_res_terminal_port+' ' + str(C) + "\n"
                lib += 'BL B1 '+main_res_terminal_port+' V=V(K_L)*V(L)' + "\n"

                #'test' inductor
                lib += 'L L 0 ' + str(L) + "\n"
                lib += 'F1 0 L BL 1' + "\n"

                ############### PROPORTIONALITY TABLES FOR MAIN ELEMENT ################################################
                lib += '* The values for the Current-Inductance-Table can be edited here:' + "\n"
                #proportionality factor for L
                lib += '* current dependent proportionality factor for L' + "\n"
                lib += 'B2 K_L 0 V=table(abs(I(BL)),{table})'.format(table=saturation_table['L']) + "\n"
                #proportionality factor for R_Fe
                lib += '* current dependent proportionality factor for R_Fe' + "\n"
                lib += 'B3 K_FE 0 V=table(abs(I(BL)),{table})'.format(table=saturation_table['R_Fe']) + "\n"

                lib += '.ENDS {inductor}'.format(inductor=model_name) + "\n"


            case constants.El.CAPACITOR:

                # define the name of the model here:
                model_name = "C_1"

                # main element parameters
                C = out['C'].value*config.CAPUNIT
                Ls = out['L'].value*config.INDUNIT
                R_s = out['R_s'].value
                R_iso = out['R_iso'].value

                lib = '* Netlist for Capacitor Model {name} (C={value}F)\n' \
                      '* Including {number} Parallely Chained Serial Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                    value=str(C),
                                                                                                    number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                ############### HIGHER ORDER ELEMENTS ##################################################################
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value*config.CAPUNIT
                    Lx = out['L%s' % circuit].value*config.INDUNIT
                    Rx = out['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'

                    #current dependent coil for higher order res:
                    lib += 'BL{no} PORT1 NL{node1} '.format(no=circuit, node1=circuit) + 'V=V(VL{no})*V(K_L{no})'.format(no=circuit) + "\n"
                    lib += 'L{no} VL{no} 0 '.format(no=circuit) + str(Lx) + "\n"
                    lib += 'FL{no} 0 VL{no} BL{no} 1'.format(no=circuit) + "\n"

                    lib += '* current dependent proportionality factor for L{no}'.format(no=circuit) + "\n"
                    lib += 'BLK{no} K_L{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no=circuit, table=saturation_table['L%s' % circuit]) + "\n"
                    lib += "\n"

                    # current dependent cap for higher order res:
                    lib += 'BC{no} NL{node1} NC{node1} '.format(no=circuit, node1=circuit) + 'I=-I(BCT{no})*V(K_C{no})'.format(no=circuit) + "\n"
                    lib += 'C{no} VC{no} 0 '.format(no=circuit) + str(Cx) + "\n"
                    lib += 'BCT{no} VC{no} 0 '.format(no=circuit) + 'V=V(NL{node1})-V(NC{node1})'.format(node1=circuit) + "\n"

                    lib += '* current dependent proportionality factor for C{no}'.format(no=circuit) + "\n"
                    lib += 'BCK{no} K_C{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no=circuit, table=saturation_table['C%s' % circuit]) + "\n"
                    lib += "\n"

                    #current dependent resistor
                    lib += 'R_{no} NC{node1} PORT2 R = limit({lo}, {hi}, {R_x} * V(K_R{no}))'.format(no=circuit, node1=circuit, lo=Rx * 1e-8, hi=Rx * 1e8,R_x=Rx) + "\n"

                    lib += '* current dependent proportionality factor for R{no}'.format(no=circuit) + "\n"
                    lib += 'BRK{no} K_R{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no=circuit, table=saturation_table['R%s' % circuit]) + "\n"
                    lib += "\n"

                ############### ACOUSTIC RESONANCE FOR MLCCs ###########################################################
                if captype == constants.captype.MLCC:
                    RA = out['R_A'].value
                    LA = out['L_A'].value*config.INDUNIT
                    CA = out['C_A'].value*config.CAPUNIT
                    #current dependent coil for higher order res:
                    lib += 'BL{no} PORT1 NL{node1} '.format(no='A', node1='A') + 'V=V(VL{no})*V(K_L{no})'.format(no='A') + "\n"
                    lib += 'L{no} VL{no} 0 '.format(no='A') + str(LA) + "\n"
                    lib += 'FL{no} 0 VL{no} BL{no} 1'.format(no='A') + "\n"

                    lib += '* current dependent proportionality factor for L{no}'.format(no='A') + "\n"
                    lib += 'BLK{no} K_L{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A', table=saturation_table['L_A']) + "\n"
                    lib += "\n"

                    # current dependent cap for higher order res:
                    lib += 'BC{no} NL{node1} NC{node1} '.format(no='A', node1='A') + 'I=-I(BCT{no})*V(K_C{no})'.format(no='A') + "\n"
                    lib += 'C{no} VC{no} 0 '.format(no='A') + str(CA) + "\n"
                    lib += 'BCT{no} VC{no} 0 '.format(no='A') + 'V=V(NL{node1})-V(NC{node1})'.format(node1='A') + "\n"

                    lib += '* current dependent proportionality factor for C{no}'.format(no='A') + "\n"
                    lib += 'BCK{no} K_C{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A', table=saturation_table['C_A']) + "\n"

                    # current dependent resistor
                    lib += 'R_{no} NC{node1} PORT2 R = limit({lo}, {hi}, {R_x} * V(K_R{no}))'.format(no='A', node1='A',lo=RA * 1e-8,hi=RA * 1e8,R_x=RA) + "\n"

                    lib += '* current dependent proportionality factor for R{no}'.format(no='A') + "\n"
                    lib += 'BRK{no} K_R{no} 0 V=table(abs(V(PORT1)-V(PORT2)),{table})'.format(no='A',table=saturation_table['R_A']) + "\n"
                    lib += "\n"

                ############### MAIN ELEMENT ###########################################################################
                lib += 'L_s LsRs Vcap ' + str(Ls) + "\n"
                lib += 'R_iso Vcap PORT2 ' + str(R_iso) + "\n"

                lib += 'B1 PORT2 Vcap I=I(E1)*V(K_C) ' + "\n"
                lib += 'E1 VC 0 Vcap PORT2 1 ' + "\n"
                lib += 'C VC 0 ' + str(C) + "\n"

                ############### PROPORTIONALITY TABLES FOR MAIN ELEMENT ################################################
                lib += 'R_p PORT1 LsRs R = limit({lo}, {hi}, {R_s} * V(K_Rs))'.format(lo=R_s * 1e-8, hi=R_s * 1e8,
                                                                                 R_s=R_s) + "\n"

                lib += '* The values for the Voltage-Capacitance-Table can be edited here:' + "\n"
                lib += 'B2 K_C 0 V=table(abs(V(PORT1)-V(PORT2)),{table}) '.format(table=saturation_table['C']) + "\n"

                lib += '* The values for the Voltage-Resistance-Table can be edited here:' + "\n"
                lib += 'B3 K_Rs 0 V=table(abs(V(PORT1)-V(PORT2)),{table}) '.format(table=saturation_table['R_s']) + "\n"

                lib += '.ENDS {name}'.format(name=model_name) + "\n"


        #create paths for the output folder and get the filename without extension
        out_path = os.path.split(self.outpath)[0]
        dir_name = os.path.normpath(self.outpath).split(os.sep)[-2]
        out_folder = os.path.join(out_path, "fit_results_%s" % dir_name)

        #create the folder; should not be necessary to handle an exception; however folder could be write protected
        try:
            os.makedirs(out_folder, exist_ok = True)
        except Exception:
            raise

        file = open(os.path.join(out_folder, "LT_Spice_Model_" + dir_name + ".lib"), "w+")
        file.write(lib)
        file.close()

    def export_parameters(self, param_array, order, fit_type, captype):
        """
        Method to output the obtained model parameters as an .xlsx file to the directory in IOhandlers output path.

        :param param_array: An array containing Parameters() type objects for each file
        :param order: The order of the model, i.e. the number of resonance circuits
        :param fit_type: Whether the model is for a coil or capacitor
        :param captype: The type of capacitor. Can be GENERIC or MLCC
        :return: None
        """

        out_dict = {}

        #write the main resonance parameters to the dict
        match fit_type:
            case constants.El.INDUCTOR:
                R_s_list = []
                R_Fe_list =[]
                L_list = []
                C_list = []
                for param_set in param_array:
                    R_s_list.append(param_set['R_s'].value)
                    R_Fe_list.append(param_set['R_Fe'].value)
                    L_list.append(param_set['L'].value*config.INDUNIT)
                    C_list.append(param_set['C'].value*config.CAPUNIT)

                out_dict['R_s'] = R_s_list
                out_dict['R_Fe'] = R_Fe_list
                out_dict['L'] = L_list
                out_dict['C'] = C_list


            case constants.El.CAPACITOR:
                R_s_list = []
                R_Iso_list = []
                L_list = []
                C_list = []
                for param_set in param_array:
                    R_s_list.append(param_set['R_s'].value)
                    R_Iso_list.append(param_set['R_iso'].value)
                    L_list.append(param_set['L'].value*config.INDUNIT)
                    C_list.append(param_set['C'].value*config.CAPUNIT)

                out_dict['R_s'] = R_s_list
                out_dict['R_iso'] = R_Iso_list
                out_dict['L'] = L_list
                out_dict['C'] = C_list

                if captype == constants.captype.MLCC:
                    R_A_list = []
                    L_A_list = []
                    C_A_list = []
                    for param_set in param_array:
                        R_A_list.append(param_set['R_A'].value)
                        L_A_list.append(param_set['L_A'].value*config.INDUNIT)
                        C_A_list.append(param_set['C_A'].value*config.CAPUNIT)
                    out_dict['R_A'] = R_A_list
                    out_dict['L_A'] = L_A_list
                    out_dict['C_A'] = C_A_list





        for key  in range(1,order+1):
            #generate key numbers and empty lists for the parameters
            C_key = "C%s" % key
            L_key = "L%s" % key
            R_key = "R%s" % key
            w_key = "w%s" % key
            BW_key = "BW%s" % key

            clist = []
            llist = []
            rlist = []
            wlist = []
            bwlist = []

            #iterate through parameter sets
            for param_set in param_array:
                clist.append(param_set[C_key].value*config.CAPUNIT)
                llist.append(param_set[L_key].value*config.INDUNIT)
                rlist.append(param_set[R_key].value)
                wlist.append(param_set[w_key].value*config.FUNIT)
                bwlist.append(param_set[BW_key].value*config.FUNIT)

            out_dict[C_key] = clist
            out_dict[L_key] = llist
            out_dict[R_key] = rlist
            out_dict[w_key] = wlist
            out_dict[BW_key] = bwlist

        #write parameters to a pandas dataframe and transpose
        data_out = pd.DataFrame(out_dict)
        # data_out.transpose()

        out_path = os.path.split(self.outpath)[0]
        dir_name = os.path.normpath(self.outpath).split(os.sep)[-2]
        out_folder = os.path.join(out_path, "fit_results_%s" % dir_name)
        try:
            os.makedirs(out_folder, exist_ok = True)
        except Exception:
            raise

        data_out.to_excel(os.path.join(out_folder, "Parameters_" + dir_name + ".xlsx"))

    def output_plot(self,freq, z21, mag, ang, mdl, filename):
        """
        Method to output a Bode-plot and a linear difference plot of the model.

        :param freq: The frequency vector
        :param z21: The measured impedance data
        :param mag: The measured, smoothed magnitude data
        :param ang: The measured, smoothed phase data
        :param mdl: The model data (complex)
        :param filename: The name of the file that the plot will be made for
        :return: None
        """

        out_path = os.path.split(self.outpath)[0]
        dir_name = os.path.normpath(self.outpath).split(os.sep)[-2]
        out_folder = os.path.join(out_path, "fit_results_%s" % dir_name)
        plot_folder = os.path.join(out_folder, "plots")


        try:
            os.makedirs(out_folder, exist_ok=True)
        except Exception:
            raise

        try:
            os.makedirs(plot_folder, exist_ok=True)
        except Exception:
            raise


        title = filename
        # fig = plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots(nrows=2,ncols=1)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        #file_title = get_file_path.results + '/03_Parameter-Fitting_' + file_name + "_" + mode
        # plt.subplot(211)
        fig = plt.gcf()
        fig.suptitle(str(title), fontsize=25, fontweight="bold")
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlim([min(freq), max(freq)])
        ax[0].set_ylabel('Magnitude in \u03A9', fontsize=16)
        ax[0].set_xlabel('Frequency in Hz', fontsize=16)
        ax[0].grid(True, which="both")
        ax[0].tick_params(labelsize=16)
        ax[0].tick_params(labelsize=16)
        ax[0].plot(freq, abs(z21), 'r', linewidth=3, alpha=0.33, label='Measured Data')
        ax[0].plot(freq, mag, 'r', linewidth=3, alpha=1, label='Filtered Data')
        # Plot magnitude of model in blue
        ax[0].plot(freq, abs(mdl), 'b--', linewidth=3, label='Model')
        ax[0].legend(fontsize=16)
        #Phase
        curve = np.angle(z21, deg=True)
        ax[1].set_xscale('log')
        ax[1].set_xlim([min(freq), max(freq)])
        ax[1].set_ylabel('Phase in °', fontsize=16)
        ax[1].set_xlabel('Frequency in Hz', fontsize=16)
        ax[1].grid(True, which="both")
        ax[1].set_yticks(np.arange(45 * (round(min(curve) / 45)), 45 * (round(max(curve) / 45)) + 1, 45.0))
        ax[1].tick_params(labelsize=16)
        ax[1].tick_params(labelsize=16)
        ax[1].plot(freq, np.angle(z21, deg=True), 'r', linewidth=3, zorder=-2, alpha=0.33, label='Measured Data')
        ax[1].plot(freq, ang, 'r', linewidth=3, zorder=-2, alpha=1, label='Filtered Data')
        #   Plot Phase of model in magenta
        ax[1].plot(freq, np.angle(mdl, deg=True), 'b--', linewidth=3, label='Model', zorder=-1)
        #plt.scatter(resonances_pos, np.zeros_like(resonances_pos) - 90, linewidth=3, color='green', s=200, marker="2",
        #            label='Resonances')
        ax[1].legend(fontsize=16)

        #may be obsolete
        plt.savefig(os.path.join(plot_folder, "Bode_plot_" + filename + ".png"), dpi = 300)

        if constants.SHOW_BODE_PLOTS:
            plt.show()
        else:
            plt.close(fig)

        #Diffplot
        if constants.OUTPUT_DIFFPLOTS:
            diff_data = abs(mdl)-abs(z21)
            diff_data_percent = (diff_data/abs(z21))*100
            title = filename + " (Model-Measurement)/Measurement in %"
            fig = plt.figure(figsize=(20, 20))
            plt.plot(freq, diff_data_percent, 'r', linewidth=3, alpha=1)
            plt.title((title), fontsize=25, fontweight="bold")
            plt.xscale('log')
            plt.yscale('linear')
            plt.xlim([min(freq), max(freq)])
            plt.ylabel('Error in %', fontsize=16)
            plt.xlabel('Frequency in Hz', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, which="both")
            # plt.plot(freq, diff_data_percent, 'r', linewidth=3, alpha=1)
            plt.savefig(os.path.join(plot_folder, "Diff_plot_" + filename + ".png"))

        if constants.SHOW_BODE_PLOTS:
            plt.show()
        else:
            plt.close(fig)




