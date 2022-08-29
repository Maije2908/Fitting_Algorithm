
# The IOhandler class shall be in charge of file handling. It will be instanced in the main function and shall be given
# to the GUI as an instance. It shall hold pointers to instances of the S2pFile class and shall be able to manage them
# in order to get the option to delete loaded files from the list and to load new files to the list

from sNpfile import *
import skrf as rf
import pandas as pd
import os
import skidl
from fitter import *
import fitterconstants

class IOhandler:

    def __init__(self):
        self.files = []
        self.outpath = None
        #self.number_of_files = 0 <- obsolete! info is in the list

    def set_out_path(self, path):
        self.outpath = path



    def load_file(self, path, ind_cap_cmc):

        try:
            for actual_path in path:
                ntwk = rf.Network(actual_path)
                touch = rf.Touchstone(actual_path) #I don't know if this is necessary, in payer's program it was used, but the
                #network class of skrf stores all the information necessary, so I would consider it a waste of memory

                # generate class for storing the data in and write to array
                newfile = SNpFile(ntwk, ntwk.name)
                self.files.append(newfile)
        except Exception as e:
            raise e

        return 0




    def generate_Netlist_2_port(self, parameters, fit_order, fit_type, saturation_table):

        out = parameters
        order = fit_order

        #TODO: generate main resonance here
        match fit_type:
            case fitterconstants.El.INDUCTOR:

                # define the name of the model here:
                model_name = "L_1"  # <inductor>
                L = out['L'].value
                # table for current dependent inductance; 'I1,L1,I2,L2,...'
                # example: '-0.3,<L_nom*0.4>,0,<L_nom>,0.3,<L_nom*0.4>'

                # I_L_table = saturation_table if saturation_table.count(',') else '0,' + "1.0"

                # do not change the rest of this section, as it defines the structure of the model
                lib = '* Netlist for Inductor Model {name} (L={value}H)\n' \
                      '* Including {number} Serially Chained Parallel Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                     value=str(L), number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                C = out['C'].value
                R_s = out['R_s'].value

                R_p = out['R_Fe'].value

                # node connection between resonant circuits
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value
                    Lx = out['L%s' % circuit].value
                    Rx = out['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'
                    lib += 'C{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Cx) + "\n"
                    lib += 'L{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Lx) + "\n"
                    lib += 'R{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Rx) + "\n"

                # node connections from current dependent inductor model
                lib += 'R_s PORT1 B1 ' + str(R_s) + "\n"
                lib += 'R_p B1 1 R = limit({lo}, {hi}, {R_Fe} * V(K_Fe))'.format(lo = R_p * 1e-8, hi = R_p * 1e8, R_Fe = R_p) + "\n"
                lib += 'C PORT1 1 ' + str(C) + "\n"
                lib += 'B1 B1 1 V=V(K_L)*V(L)' + "\n"

                #'test' inductor
                lib += 'L L 0 ' + str(L) + "\n"
                lib += 'F1 0 L B1 1' + "\n"


                lib += '* The values for the Current-Inductance-Table can be edited here:' + "\n"
                #   lib += '* e.g. -0.3, <L_sat>, 0, <L_nom>, 0.3, <L_sat>' + "\n"

                #proportionality factor for L
                lib += '* current dependent proportionality factor for L' + "\n"
                lib += 'B2 K_L 0 V=table(abs(I(B1)),{table})'.format(table=saturation_table['L']) + "\n"
                #proportionality factor for R_Fe
                lib += '* current dependent proportionality factor for R_Fe' + "\n"
                lib += 'B3 K_FE 0 V=table(abs(I(B1)),{table})'.format(table=saturation_table['R_Fe']) + "\n"


                lib += '.ENDS {inductor}'.format(inductor=model_name) + "\n"


            case fitterconstants.El.CAPACITOR:

                # define the name of the model here:
                model_name = "C_1"  # <inductor>
                C = out['C'].value
                # table for current dependent inductance; 'I1,L1,I2,L2,...'
                # example: '-0.3,<C_dc/C_nom>,0,1,0.3,<Cdc/C_nom>'
                I_L_table = saturation_table if saturation_table.count(',') else '0,' + "1.0"

                # do not change the rest of this section, as it defines the structure of the model
                lib = '* Netlist for Capacitor Model {name} (C={value}F)\n' \
                      '* Including {number} Parallely Chained Serial Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                    value=str(C),
                                                                                                    number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                Ls = out['L'].value
                R_s = out['R_s'].value
                R_iso = out['R_iso'].value

                # node connections from voltage dependent capacitor model
                lib += 'R_s PORT1 LsRs ' + str(R_s) + "\n"
                lib += 'R_iso Vcap 0 ' + str(R_iso) + "\n"
                lib += '* The values for the Voltage-Capacitance-Table can be edited here:' + "\n"
                lib += 'B2 K 0 V=table(abs(V(PORT1)),{table}) '.format(table=I_L_table) + "\n"
                lib += 'L_s LsRs Vcap ' + str(Ls) + "\n"
                lib += 'E1 E1 0 Vcap 0 1 ' + "\n"
                lib += 'C E1 0 ' + str(C) + "\n"
                lib += 'B1 0 Vcap I=I(E1)*V(K) ' + "\n"

                # node connection between resonant circuits
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value
                    Lx = out['L%s' % circuit].value
                    Rx = out['R%s' % circuit].value

                    lib += 'R{no} PORT1 {node2} '.format(no=circuit, node2=circuit) + str(Rx) + "\n"
                    lib += 'L{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=order + circuit) + str(
                        Lx) + "\n"
                    lib += 'C{no} {node1} 0 '.format(no=circuit, node1=order + circuit) + str(Cx) + "\n"

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


    def generate_Netlist_2_port_full_fit(self, parameters, fit_order, fit_type, saturation_table, captype=None):

        out = parameters
        order = fit_order

        #TODO: generate main resonance here
        match fit_type:
            case fitterconstants.El.INDUCTOR:

                # define the name of the model here:
                model_name = "L_1"  # <inductor>
                L = out['L'].value
                # table for current dependent inductance; 'I1,L1,I2,L2,...'
                # example: '-0.3,<L_nom*0.4>,0,<L_nom>,0.3,<L_nom*0.4>'

                # I_L_table = saturation_table if saturation_table.count(',') else '0,' + "1.0"

                # do not change the rest of this section, as it defines the structure of the model
                lib = '* Netlist for Inductor Model {name} (L={value}H)\n' \
                      '* Including {number} Serially Chained Parallel Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                     value=str(L), number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                C = out['C'].value
                R_s = out['R_s'].value

                R_p = out['R_Fe'].value

                # node connection between resonant circuits
                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value
                    Lx = out['L%s' % circuit].value
                    Rx = out['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'

                    #current dependent coil for higher order res:
                    lib += 'BL{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + 'V=V(VL{no})*V(K_L{no})'.format(no=circuit) + "\n"
                    lib += 'L{no} VL{no} 0 '.format(no=circuit) + str(Lx) + "\n"
                    lib += 'FL{no} 0 VL{no} BL{no} 1'.format(no=circuit) + "\n"

                    lib += '* current dependent proportionality factor for L{no}'.format(no=circuit) + "\n"
                    lib += 'BLK{no} K_L{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['L%s' % circuit]) + "\n"
                    lib += "\n"

                    # current dependent cap for higher order res:
                    lib += 'BC{no} {node1} {node2} '.format(no=circuit, node1=circuit,node2=node2) + 'I=-I(BCT{no})*V(K_C{no})'.format(no=circuit) + "\n"
                    lib += 'C{no} VC{no} 0 '.format(no=circuit) + str(Cx) + "\n"
                    lib += 'BCT{no} VC{no} 0 '.format(no=circuit) + 'V=V({node1})-V({node2})'.format(node1=circuit, node2=node2) + "\n"

                    lib += '* current dependent proportionality factor for C{no}'.format(no=circuit) + "\n"
                    lib += 'BCK{no} K_C{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['C%s' % circuit]) + "\n"
                    lib += "\n"

                    #current dependent resistor
                    lib += 'R_{no} {node1} {node2} R = limit({lo}, {hi}, {R_x} * V(K_R{no}))'.format(no=circuit, node1=circuit,node2=node2, lo=Rx * 1e-8, hi=Rx * 1e8,R_x=Rx) + "\n"

                    lib += '* current dependent proportionality factor for R{no}'.format(no=circuit) + "\n"
                    lib += 'BRK{no} K_R{no} 0 V=table(abs(I(BL)),{table})'.format(no=circuit, table=saturation_table['R%s' % circuit]) + "\n"
                    lib += "\n"

                # node connections from current dependent inductor model
                lib += 'R_s PORT1 B1 ' + str(R_s) + "\n"
                lib += 'R_p B1 1 R = limit({lo}, {hi}, {R_Fe} * V(K_Fe))'.format(lo = R_p * 1e-8, hi = R_p * 1e8, R_Fe = R_p) + "\n"
                lib += 'C PORT1 1 ' + str(C) + "\n"
                lib += 'BL B1 1 V=V(K_L)*V(L)' + "\n"

                #'test' inductor
                lib += 'L L 0 ' + str(L) + "\n"
                lib += 'F1 0 L BL 1' + "\n"


                lib += '* The values for the Current-Inductance-Table can be edited here:' + "\n"
                #   lib += '* e.g. -0.3, <L_sat>, 0, <L_nom>, 0.3, <L_sat>' + "\n"

                #proportionality factor for L
                lib += '* current dependent proportionality factor for L' + "\n"
                lib += 'B2 K_L 0 V=table(abs(I(BL)),{table})'.format(table=saturation_table['L']) + "\n"
                #proportionality factor for R_Fe
                lib += '* current dependent proportionality factor for R_Fe' + "\n"
                lib += 'B3 K_FE 0 V=table(abs(I(BL)),{table})'.format(table=saturation_table['R_Fe']) + "\n"


                lib += '.ENDS {inductor}'.format(inductor=model_name) + "\n"


            case fitterconstants.El.CAPACITOR:

                # define the name of the model here:
                model_name = "C_1"  # <cap>
                C = out['C'].value

                # do not change the rest of this section, as it defines the structure of the model
                lib = '* Netlist for Capacitor Model {name} (C={value}F)\n' \
                      '* Including {number} Parallely Chained Serial Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                    value=str(C),
                                                                                                    number=order)

                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'


                for circuit in range(1, order + 1):
                    Cx = out['C%s' % circuit].value
                    Lx = out['L%s' % circuit].value
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

                #acoustic resonance
                if captype == fitterconstants.captype.MLCC:
                    RA = out['R_A'].value
                    LA = out['L_A'].value
                    CA = out['C_A'].value
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





                Ls = out['L'].value
                C = out['C'].value
                R_s = out['R_s'].value
                R_iso = out['R_iso'].value

                # node connections from voltage dependent capacitor model
                # lib += 'R_s PORT1 LsRs ' + str(R_s) + "\n"
                lib += 'L_s LsRs Vcap ' + str(Ls) + "\n"
                lib += 'R_iso Vcap PORT2 ' + str(R_iso) + "\n"

                lib += 'B1 PORT2 Vcap I=I(E1)*V(K_C) ' + "\n"
                lib += 'E1 VC 0 Vcap PORT2 1 ' + "\n"
                lib += 'C VC 0 ' + str(C) + "\n"

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


    def export_parameters(self, param_array, order, fit_type):

        out_dict = {}


        #write the main resonance parameters to the dict
        match fit_type:
            case fitterconstants.El.INDUCTOR:
                R_s_list = []
                R_Fe_list =[]
                L_list = []
                C_list = []
                for param_set in param_array:
                    R_s_list.append(param_set['R_s'].value)
                    R_Fe_list.append(param_set['R_Fe'].value)
                    L_list.append(param_set['L'].value)
                    C_list.append(param_set['C'].value)

                out_dict['R_s'] = R_s_list
                out_dict['R_Fe'] = R_Fe_list
                out_dict['L'] = L_list
                out_dict['C'] = C_list


            case fitterconstants.El.CAPACITOR:
                R_s_list = []
                R_Iso_list = []
                L_list = []
                C_list = []
                for param_set in param_array:
                    R_s_list.append(param_set['R_s'].value)
                    R_Iso_list.append(param_set['R_iso'].value)
                    L_list.append(param_set['L'].value)
                    C_list.append(param_set['C'].value)

                out_dict['R_s'] = R_s_list
                out_dict['R_iso'] = R_Iso_list
                out_dict['L'] = L_list
                out_dict['C'] = C_list




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
                clist.append(param_set[C_key].value)
                llist.append(param_set[L_key].value)
                rlist.append(param_set[R_key].value)
                wlist.append(param_set[w_key].value)
                bwlist.append(param_set[BW_key].value)

            out_dict[C_key] = clist
            out_dict[L_key] = llist
            out_dict[R_key] = rlist
            out_dict[w_key] = wlist
            out_dict[BW_key] = bwlist

        #write parameters to a pandas dataframe and transpose
        data_out = pd.DataFrame(out_dict)
        data_out.transpose()

        out_path = os.path.split(self.outpath)[0]
        dir_name = os.path.normpath(self.outpath).split(os.sep)[-2]
        out_folder = os.path.join(out_path, "fit_results_%s" % dir_name)
        try:
            os.makedirs(out_folder, exist_ok = True)
        except Exception:
            raise

        data_out.to_excel(os.path.join(out_folder, "Parameters_" + dir_name + ".xlsx"))



    def output_plot(self,freq, z21, mag, ang, mdl, filename):

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
        fig = plt.figure(figsize=(20, 20))
        #file_title = get_file_path.results + '/03_Parameter-Fitting_' + file_name + "_" + mode
        plt.subplot(211)
        plt.title(str(title), pad=20, fontsize=25, fontweight="bold")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([min(freq), max(freq)])
        plt.ylabel('Magnitude in \u03A9', fontsize=16)
        plt.xlabel('Frequency in Hz', fontsize=16)
        plt.grid(True, which="both")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(freq, abs(z21), 'r', linewidth=3, alpha=0.33, label='Measured Data')
        plt.plot(freq, mag, 'r', linewidth=3, alpha=1, label='Filtered Data')
        # Plot magnitude of model in blue
        plt.plot(freq, abs(mdl), 'b--', linewidth=3, label='Model')
        plt.legend(fontsize=16)

        #Phase
        curve = np.angle(z21, deg=True)

        plt.subplot(212)
        plt.xscale('log')
        plt.xlim([min(freq), max(freq)])
        plt.ylabel('Phase in °', fontsize=16)
        plt.xlabel('Frequency in Hz', fontsize=16)
        plt.grid(True, which="both")
        plt.yticks(np.arange(45 * (round(min(curve) / 45)), 45 * (round(max(curve) / 45)) + 1, 45.0))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(freq, np.angle(z21, deg=True), 'r', linewidth=3, zorder=-2, alpha=0.33, label='Measured Data')
        plt.plot(freq, ang, 'r', linewidth=3, zorder=-2, alpha=1, label='Filtered Data')
        #   Plot Phase of model in magenta
        plt.plot(freq, np.angle(mdl, deg=True), 'b--', linewidth=3, label='Model', zorder=-1)
        #plt.scatter(resonances_pos, np.zeros_like(resonances_pos) - 90, linewidth=3, color='green', s=200, marker="2",
        #            label='Resonances')
        plt.legend(fontsize=16)

        plt.savefig(os.path.join(plot_folder, "Bode_plot_" + filename + ".png"))

        if fitterconstants.SHOW_BODE_PLOTS:
            plt.show()
        else:
            plt.close(fig)

        #Diffplot
        if fitterconstants.OUTPUT_DIFFPLOTS:
            diff_data = abs(mdl)-abs(z21)
            diff_data_percent = (diff_data/abs(z21))*100
            title = filename + " (Model-Measurement)/Measurement in %"
            fig = plt.figure(figsize=(20, 20))
            plt.title(str(title), pad=20, fontsize=25, fontweight="bold")
            plt.xscale('log')
            plt.yscale('linear')
            plt.xlim([min(freq), max(freq)])
            plt.ylabel('Error in %', fontsize=16)
            plt.xlabel('Frequency in Hz', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, which="both")
            plt.plot(freq, diff_data_percent, 'r', linewidth=3, alpha=1)
            plt.savefig(os.path.join(plot_folder, "Diff_plot_" + filename + ".png"))

        if fitterconstants.SHOW_BODE_PLOTS:
            plt.show()
        else:
            plt.close(fig)

        # plt.close(fig)



