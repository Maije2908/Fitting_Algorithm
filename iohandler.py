
# The IOhandler class shall be in charge of file handling. It will be instanced in the main function and shall be given
# to the GUI as an instance. It shall hold pointers to instances of the S2pFile class and shall be able to manage them
# in order to get the option to delete loaded files from the list and to load new files to the list

from sNpfile import *
import skrf as rf
import skidl
from fitter import *
import fitterconstants

class IOhandler:

    def __init__(self):
        self.files = []
        #self.number_of_files = 0 <- obsolete! info is in the list


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

    def generate_Netlist_2_port(self, fitterinstance: Fitter, fit_type, path, I_L_table_input):

        out = fitterinstance.out
        order = fitterinstance.order

        #TODO: generate main resonance here
        match fit_type:
            case fitterconstants.El.INDUCTOR:

                # define the name of the model here:
                model_name = "L_1"  # <inductor>
                L = out.params['L'].value
                # table for current dependent inductance; 'I1,L1,I2,L2,...'
                # example: '-0.3,<L_nom*0.4>,0,<L_nom>,0.3,<L_nom*0.4>'

                # TODO: look into I_L_table
                I_L_table = I_L_table_input if I_L_table_input.count(',') else '0,' + "1.0"

                # do not change the rest of this section, as it defines the structure of the model
                lib = '* Netlist for Inductor Model {name} (L={value}H)\n' \
                      '* Including {number} Serially Chained Parallel Resonant Circuits\n*\n'.format(name=model_name,
                                                                                                     value=str(L), number=order)
                lib += '.SUBCKT {name} PORT1 PORT2'.format(name=model_name) + '\n*\n'

                C = out.params['C'].value
                R_s = out.params['R_s'].value

                R_p = out.params['R_Fe'].value

                # node connection between resonant circuits
                for circuit in range(1, order + 1):
                    Cx = out.params['C%s' % circuit].value
                    Lx = out.params['L%s' % circuit].value
                    Rx = out.params['R%s' % circuit].value
                    node2 = circuit + 1 if circuit < order else 'PORT2'
                    lib += 'C{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Cx) + "\n"
                    lib += 'L{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Lx) + "\n"
                    lib += 'R{no} {node1} {node2} '.format(no=circuit, node1=circuit, node2=node2) + str(Rx) + "\n"

                # node connections from current dependent inductor model
                lib += 'R_s PORT1 B1 ' + str(R_s) + "\n"
                lib += 'R_p B1 1 ' + str(R_p) + "\n"
                lib += 'C 1 PORT1 ' + str(C) + "\n"
                lib += 'L L 0 ' + str(L) + "\n"
                lib += 'B1 B1 1 V=V(inductance)*V(L)' + "\n"
                lib += 'F1 0 L B1 1' + "\n"
                lib += '* The values for the Current-Inductance-Table can be edited here:' + "\n"
                #   lib += '* e.g. -0.3, <L_sat>, 0, <L_nom>, 0.3, <L_sat>' + "\n"
                #TODO: I_L_Table again
                lib += 'B2 inductance 0 V=table(I(B1),{table})'.format(table=I_L_table) + "\n"
                lib += '.ENDS {inductor}'.format(inductor=model_name) + "\n"

                pass

            case fitterconstants.El.CAPACITOR:
                #TODO: adapt for capacitors
                pass
        #TODO: adapt for linux
        file_name = path.split("\\")[-1][:-4]
        out_path = '\\'.join(path.split("\\")[:-2])
        #write to file
        f2 = open(out_path + "/05_LTspice_" + file_name
                  + ".lib", "w+")
        f2.write(lib)

        f2.close()



