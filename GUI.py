# import packages
import tkinter as tk
from tkinter import filedialog
#maybe import later for logging to gui
#import tkinter.scrolledtext as scroll_text
#import logging
import fitterconstants
from fitter import *
import config
import copy
import os
import re
from tkinter import scrolledtext
from texthandler import *
from lmfit import Parameters
import multiprocessing as mp

'''
***********************************************************************************************************************
CREATE GUI:
***********************************************************************************************************************

Here the GUI is created.

https://www.inf-schule.de/software/gui/entwicklung_tkinter/layout/pack

NOTE: There are hardly any error-messages for wrong filetypes ect. The GUI should make the use of the program easier.

***********************************************************************************************************************
'''


class GUI:
    def __init__(self, iohandler_instance):
        # declare instance variables
        self.st = None
        self.texthndl = None
        self.entry_saturation = None
        self.entry_nominal_value = None
        self.entry_resistance = None
        self.entry_prominence = None
        self.shunt_series = None
        self.selected_s2p_files = None
        self.iohandler = iohandler_instance
        self.fitter = None
        self.logger = None
        # variables for the file list
        self.filelist_frame = None
        self.filename_label = []
        self.filename_entry = []
        self.filename_ref_button = []
        self.ref_file_select =None

        # Window config
        self.root: tk.Tk = tk.Tk()
        self.root.wm_title('Fitting Program V2')
        self.root.config(bg='#FFFFFF')

        # print screen size
        print("Width: ", self.root.winfo_screenwidth())
        print("Height: ", self.root.winfo_screenheight())

        # set window size
        self.root.geometry(
            "%dx%d" % (int(0.5 * self.root.winfo_screenwidth()), int(0.75 * self.root.winfo_screenheight())))

        # here starts the creation of the widgets
        self.create_drop_down()
        self.create_specification_field()
        self.create_browse_button()
        self.create_run_button()
        self.create_log_window()
        self.create_shunt_series_radio_button()
        self.create_filelist_frame()
        self.create_clear_files_button()
        # self.create_file_list()

    def create_drop_down(self):
        self.drop_down_var = tk.StringVar(self.root, config.DROP_DOWN_ELEMENTS[0])

        self.option_menu = tk.OptionMenu(self.root, self.drop_down_var, *config.DROP_DOWN_ELEMENTS)
        max_drop_length = len(max(config.DROP_DOWN_ELEMENTS, key=len))
        self.option_menu.config(font=config.DROP_DOWN_FONT, width=max_drop_length + 5, height=config.DROP_DOWN_HEIGHT)
        self.option_menu.grid(column=0, row=0, columnspan=2, sticky=tk.W, **config.HEADLINE_PADDING)

    def create_shunt_series_radio_button(self):
        self.shunt_series = tk.IntVar()
        label_calc = tk.Label(self.root, text="Z Calculation Method")
        label_calc.config(font=config.ENTRY_FONT)
        label_calc.grid(column=2, row=1, sticky=tk.NW, **config.HEADLINE_PADDING)

        r1 = tk.Radiobutton(self.root, text = 'Shunt Through', variable=self.shunt_series,value=config.SHUNT_THROUGH)
        r2 = tk.Radiobutton(self.root, text = 'Series Through', variable=self.shunt_series,value=config.SERIES_THROUGH)
        r1.grid(column=2, row=2)
        r2.grid(column=2, row=3)


    def create_specification_field(self):

        # validate command for inputs "register" is necessary so that the actual input is checked (would otherwise
        # update after input)
        vcmd = (self.root.register(self.entry_number_callback), "%P")

        # Headline
        label_spec = tk.Label(self.root, text="Specification", bg=config.BCKGND_COLOR)
        label_spec.config(font=config.HEADLINE_FONT)
        label_spec.grid(column=0, row=1, columnspan=2, sticky=tk.NW, **config.HEADLINE_PADDING)

        # initial value
        passive_element_label = tk.Label(self.root, text="F/C", bg=config.BCKGND_COLOR)
        passive_element_label.config(font=config.ENTRY_FONT)
        passive_element_label.grid(column=0, row=2, sticky=tk.W, **config.SPEC_PADDING)

        self.entry_nominal_value = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_nominal_value.config(font=config.ENTRY_FONT)
        self.entry_nominal_value.grid(column=1, row=2, sticky=tk.W, **config.ENTRY_PADDING)

        # initial resistance value
        label_resistance = tk.Label(self.root, text="\u03A9", bg=config.BCKGND_COLOR)
        label_resistance.config(font=config.ENTRY_FONT)
        label_resistance.grid(column=0, row=3, sticky=tk.W, **config.SPEC_PADDING)

        self.entry_resistance = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_resistance.config(font=config.ENTRY_FONT)
        self.entry_resistance.grid(column=1, row=3, sticky=tk.W, **config.ENTRY_PADDING)

        # Saturation Table
        label_saturation = tk.Label(self.root, text="Saturation Table", bg=config.BCKGND_COLOR)
        label_saturation.config(font=config.ENTRY_FONT)
        label_saturation.grid(column=0, row=4, sticky=tk.W, **config.SPEC_PADDING)
        # endregion

        self.entry_saturation = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_saturation.config(font=config.ENTRY_FONT)
        self.entry_saturation.grid(column=1, row=4, sticky=tk.W, **config.ENTRY_PADDING)

        # Prominence
        label_prominence = tk.Label(self.root, text="Prominence in dB", bg=config.BCKGND_COLOR)
        label_prominence.config(font=config.ENTRY_FONT)
        label_prominence.grid(column=0, row=5, sticky=tk.W, **config.SPEC_PADDING)

        self.entry_prominence = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_prominence.config(font=config.ENTRY_FONT)
        self.entry_prominence.grid(column=1, row=5, sticky=tk.W, **config.ENTRY_PADDING)

    def create_browse_button(self):
        browse_button = tk.Button(self.root, command=self.callback_browse_s2p_file, text="Select s2p File(s)")
        browse_button.config(font=config.ENTRY_FONT)
        browse_button.grid(column=0, row=6, sticky=tk.W, **config.BUTTON_LEFT_PADDING)

    def create_run_button(self):
        browse_button = tk.Button(self.root, command=self.callback_run, text="Run")
        browse_button.config(font=config.ENTRY_FONT)
        browse_button.grid(column=1, row=6, sticky=tk.W, **config.BUTTON_RIGHT_PADDING)

    def create_clear_files_button(self):
        browse_button = tk.Button(self.root, command=self.callback_clear_files, text="Clear Files")
        browse_button.config(font=config.ENTRY_FONT)
        browse_button.grid(column=4, row=0, sticky=tk.W, **config.BUTTON_RIGHT_PADDING)

    def create_log_window(self):
        self.st = scrolledtext.ScrolledText(self.root, state='disabled')#, width=config.LOG_WIDTH,  height=config.LOG_HEIGHT)
        self.st.configure(font='TkFixedFont')
        self.st.grid(column=0,row=9,columnspan=3,sticky=tk.W,**config.ENTRY_PADDING)
        # self.st.pack()
        self.texthndl = Text_Handler(self.st)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.texthndl)
        self.logger.setLevel(logging.INFO)

    def create_filelist_frame (self):
        self.filelist_frame = tk.LabelFrame(self.root, text = 'Files')
        self.filelist_frame.grid(column = 4, row = 1, rowspan = 10, columnspan=2, sticky=tk.NW, **config.SPEC_PADDING)
        # create headings for the columns
        ref_lbl = tk.Label(self.filelist_frame, text='Reference File?')
        name_lbl = tk.Label(self.filelist_frame, text='Filename')
        cond_lbl = tk.Label(self.filelist_frame, text='Current/Voltage')
        ref_lbl.grid(column=0,row=0)
        name_lbl.grid(column=1,row=0)
        cond_lbl.grid(column=2,row=0)
        #create an integer variable for the radiobuttons in order to select the reference file
        self.ref_file_select = tk.IntVar()

    def get_file_current_voltage_values(self):

        file_current_voltage_list = []

        for entry in self.filename_entry:
            file_current_voltage_list.append(self.entry_to_float(entry.get()))

        return file_current_voltage_list


    def update_file_list(self):
        rownumber = len(self.filename_label) + 1
        vcmd = (self.root.register(self.entry_number_callback), "%P")

        for file in self.iohandler.files:

            #create a label for the filename
            label_name = file.name
            label = tk.Label(self.filelist_frame, text= label_name)
            entry = tk.Entry(self.filelist_frame, width = 5, validate='all', validatecommand=(vcmd))
            label.grid(column=1, row = rownumber, sticky=tk.NW, **config.SPEC_PADDING)
            entry.grid(column=2, row = rownumber, sticky=tk.NSEW, **config.SPEC_PADDING)
            #create a button for the selection of the reference file
            r_button = tk.Radiobutton(self.filelist_frame, variable=self.ref_file_select, value =rownumber - 1)
            r_button.grid(column=0, row=rownumber)

            rownumber += 1
            self.filename_entry.append(entry)
            self.filename_label.append(label)
            self.filename_ref_button.append(r_button)


    def callback_clear_files(self):
        #method to clear the file list and also the files from the iohandler
        self.iohandler.files = []
        for label in self.filename_label:
            label.destroy()
        for entry in self.filename_entry:
            entry.destroy()
        for r_button in self.filename_ref_button:
            r_button.destroy()
        #
        self.filename_label = []
        self.filename_entry = []
        self.filename_ref_button = []



    ####################################################################################################################
    # Button commands

    # Method that is invoked when the "open file" button is pressed; opens a file dialoge, and invokes the IOhandler
    # in order to load the touchstone file
    def callback_browse_s2p_file(self):
        filename = tk.filedialog.askopenfilename(title=
                                                 'Open Measured Data (Touchstone-Format)',
                                                 filetypes=((".s2p", "*.s2p*"),
                                                            ("all files", "*.*")), multiple=True)

        path_list = [None] * len(filename)
        for file_number in range(len(path_list)):
            path_list[file_number] = os.path.abspath(filename[file_number])

        # set instance variable for s2p files to selected files
        # EDIT: this might become obsolete since the iohandler loads the files directly
        self.selected_s2p_files = path_list

        #NOTE: second parameter should be to select inductivity/capacitance; unsure (yet) if this is necessary
        try:
            self.iohandler.load_file(path_list, 2)
            self.logger.info("Opened Files:")
            [self.logger.info(file_path) for file_path in path_list]
        except Exception as e:
            self.logger.error("ERROR: There was an error, opening one of the selected files:")
            self.logger.error(str(e))

        #insert the files to the listbox
        self.update_file_list()

        return 0


    #method to run the fitting algorithm, invoked when "run" button is pressed
    def callback_run(self):

        # TODO: the capacitor type is hardcoded here, consider some entry box or something
        captype = fitterconstants.captype.MLCC


        self.logger.info("----------Run----------\n")



        #get values from the entry boxes
        passive_nom = self.entry_to_float(self.entry_nominal_value.get())
        res         = self.entry_to_float(self.entry_resistance.get())
        prom        = self.entry_to_float(self.entry_prominence.get())
        sat         = self.entry_to_float(self.entry_saturation.get())

        #get the shunt/series through setting
        shunt_series = self.shunt_series.get()

        #get the type of element from the dropdown list
        element_type_str = self.drop_down_var.get()


        try:
            #raise an exception if shunt/series through was not set
            if not (shunt_series):
                # self.logger.error("Shunt/Series-Through not set!\nPlease select a calculation mode")
                raise Exception("Shunt/Series-Through not set!\nPlease select a calculation mode")


            if element_type_str == config.DROP_DOWN_ELEMENTS[0]:
                fit_type = config.El.INDUCTOR
            elif element_type_str == config.DROP_DOWN_ELEMENTS[1]:
                fit_type = config.El.CAPACITOR
            elif element_type_str == config.DROP_DOWN_ELEMENTS[2]:
                # self.logger.error('CMCs not implemented yet, please change element type')
                raise Exception('CMCs not implemented yet, please change element type')

        except Exception as e:
            #write exception to log
            self.logger.error(str(e) + '\n')
            raise





        try:
            # create a fitter instance (the logger instance needs to be passed to the constructor)
            self.fitter = Fitter(self.logger)

            #check if files are present
            if not self.iohandler.files:
                raise Exception("Error: No Files present")


            #get selected reference file and make a list with all files that are not the reference file
            ref_file = self.iohandler.files[self.ref_file_select.get()]
            other_files = np.concatenate((self.iohandler.files[:self.ref_file_select.get()], self.iohandler.files[self.ref_file_select.get() + 1:]))
            if ref_file is None:
                raise Exception("Error: Please select a reference file")


            #get the values from the entries that define the currents/voltages of each file
            dc_bias = self.get_file_current_voltage_values()
            if None in dc_bias:
                raise Exception("Error: Please specify the current/voltage values for the given files!")


            ################ PARSING AND PRE-PROCESSING ################################################################

            #create an array for the fitter instances and one for the parameters
            fitters = []
            parameter_list = []

            for it, file in enumerate(self.iohandler.files):
                #instanciate a fitter and pass it the file and the logger instance
                fitter_instance = Fitter(self.logger)
                fitter_instance.set_file(file)

                #calculate the impedance data for the fitter
                match shunt_series:
                    case config.SHUNT_THROUGH:
                        fitter_instance.calc_shunt_thru(config.Z0)
                    case config.SERIES_THROUGH:
                        fitter_instance.calc_series_thru(config.Z0)

                #smooth the impedance data and pass the specification
                fitter_instance.smooth_data()
                fitter_instance.set_specification(passive_nom, res, prom, sat, fit_type, captype)

                #write instance to list
                fitters.append(fitter_instance)

            ################ END PARSING AND PRE-PROCESSING ############################################################

            ################ MAIN RESONANCE FIT ########################################################################

            for it, fitter in enumerate(fitters):

                params = Parameters()
                f0  = fitter.get_main_resonance()

                #create the main resonance parameters
                try:
                    main_res_params = fitter.create_nominal_parameters(params)
                except Exception:
                    raise Exception("Error: Something went wrong while trying to create nominal parameters; "
                                    "check if the element type is correct")


                if it == 0:
                    #fit the main resonance for the first file
                    match fit_type:
                        case fitterconstants.El.INDUCTOR:
                            fitted_main_res_params = fitter.fit_main_res_inductor_file_1(main_res_params)
                        case fitterconstants.El.CAPACITOR:
                            fitted_main_res_params = fitter.fit_main_res_capacitor_file_1(main_res_params)
                else:
                    #fit the main resonance for every other file (we have to overwrite some parameters here, since the
                    # main parasitic element (C for inductors, L for capacitors) and the R_s should be constrained
                    main_res_params = fitter.overwrite_main_res_params_file_n(main_res_params, parameter_list[0])
                    match fit_type:
                        case fitterconstants.El.INDUCTOR:
                            fitted_main_res_params = fitter.fit_main_res_inductor_file_n(main_res_params)
                        case fitterconstants.El.CAPACITOR:
                            fitted_main_res_params = fitter.fit_main_res_capacitor_file_n(main_res_params)
                #finally write the fitted main resonance parameters to the list
                parameter_list.append(fitted_main_res_params)

            ################ END MAIN RESONANCE FIT ####################################################################

            ################ ACOUSITC RESONANCE DETECTION FOR MLCCs ####################################################

            # get acoustic resonance frequency for all files, if not found write "None" to list
            if captype == fitterconstants.captype.MLCC and fit_type == fitterconstants.El.CAPACITOR:
                acoustic_res_frqs = []
                for fitter in fitters:
                    try:
                        acoustic_res_frqs.append(fitter.get_acoustic_resonance())
                    except:
                        acoustic_res_frqs.append(None)

                # iterate through the fitters in reversed order and fit the acoustic resonance
                if len(fitters) > 1:
                    for it, fitter in reversed(list(enumerate(fitters))):
                        if acoustic_res_frqs[it] is not None:
                            fitter.set_acoustic_resonance_frequency(acoustic_res_frqs[it])
                            parameter_list[it] = fitter.fit_acoustic_resonance(parameter_list[it])
                        else:
                            # if there is no frequency for the actual resonance take the previous frequency
                            # (NOTE): this might even be obsolete
                            acoustic_res_frqs[it] = acoustic_res_frqs[it - 1]
                            fitter.set_acoustic_resonance_frequency(acoustic_res_frqs[it])
                            # manually write the parameters of the previous fit to the dataset
                            hi_R = parameter_list[it - 1]['R_A'].value * 1e4
                            parameter_list[it].add('L_A', value=parameter_list[it - 1]['L_A'].value)
                            parameter_list[it].add('C_A', value=parameter_list[it - 1]['C_A'].value)
                            parameter_list[it].add('R_A', value=hi_R)
                else:
                    self.logger.info("WARNING: MLCCs captype selected, but only one file is present. Switching to generic captype")
                    captype = fitterconstants.captype.GENERIC
                    for fitter in fitters:
                        fitter.set_captype(captype)

            ################ END ACOUSITC RESONANCE DETECTION FOR MLCCs ################################################

            ################ HIGHER ORDER RESONANCES ###################################################################

            for it, fitter in enumerate(fitters):

                fitter.get_resonances()

                if fitter.order:
                    #generate parameter sets in two configurations 1=Q constrained; 2=free R and C
                    params1 = copy.copy(fitter.create_higher_order_parameters(1, parameter_list[it]))
                    params2 = copy.copy(fitter.create_higher_order_parameters(2, parameter_list[it]))

                    #correct obtained parameters and do the band pre-fit
                    correct_main_res = False
                    num_iterations = 4
                    params1 = fitter.correct_parameters(params1, correct_main_res, num_iterations)
                    params2 = fitter.correct_parameters(params2, correct_main_res, num_iterations)
                    params1 = fitter.pre_fit_bands(params1)
                    params2 = fitter.pre_fit_bands(params2)

                    #fit the whole curve
                    fit_params1 = fitter.fit_curve_higher_order(params1)
                    fit_params2 = fitter.fit_curve_higher_order(params2)

                    #check wich model fits best
                    out_params = fitter.select_param_set([fit_params1, fit_params2])
                    parameter_list[it] = out_params

                    #write the model data to the fitter instance
                    # fitter.write_model_data(out_params)

                    if DEBUG_FIT:
                        fitter.plot_curve(parameter_list[it], fitter.order, False, str(fitter.file.name) + ' fitted higher order resonances')


            ################ END HIGHER ORDER RESONANCES ###############################################################

            ############### MATCH PARAMETERS ###########################################################################

            parameter_list = self.match_parameters(parameter_list, fitters, captype)

            ############### END MATCH PARAMETERS #######################################################################

            ################ SATURATION TABLE(S) #######################################################################

            order = max([fitter.order for fitter in fitters])
            # saturation table for nominal value
            # create saturation table and get nominal value
            saturation_table = {}
            match fit_type:
                case fitterconstants.El.INDUCTOR:
                    saturation_table['L'] = self.generate_saturation_table(parameter_list, 'L', dc_bias)
                    saturation_table['R_Fe'] = self.generate_saturation_table(parameter_list, 'R_Fe',
                                                                              dc_bias)
                case fitterconstants.El.CAPACITOR:
                    saturation_table['C'] = self.generate_saturation_table(parameter_list, 'C', dc_bias)
                    saturation_table['R_s'] = self.generate_saturation_table(parameter_list, 'R_s', dc_bias)

            # write saturation table for acoustic resonance
            if fit_type == fitterconstants.El.CAPACITOR and captype == fitterconstants.captype.MLCC:
                saturation_table['R_A'] = self.generate_saturation_table(parameter_list, 'R_A', dc_bias)
                saturation_table['L_A'] = self.generate_saturation_table(parameter_list, 'L_A', dc_bias)
                saturation_table['C_A'] = self.generate_saturation_table(parameter_list, 'C_A', dc_bias)

            if fitterconstants.FULL_FIT:

                # create saturation tables for all parameters
                for key_number in range(1, order + 1):
                    # create keys
                    C_key = "C%s" % key_number
                    L_key = "L%s" % key_number
                    R_key = "R%s" % key_number

                    saturation_table[C_key] = self.generate_saturation_table(parameter_list, C_key, dc_bias)
                    saturation_table[L_key] = self.generate_saturation_table(parameter_list, L_key, dc_bias)
                    saturation_table[R_key] = self.generate_saturation_table(parameter_list, R_key, dc_bias)

            ################ END SATURATION TABLE(S) ###################################################################

            ################ OUTPUT ####################################################################################

            #set path for IO handler
            path_out = self.selected_s2p_files[0]
            self.iohandler.set_out_path(path_out)

            #export parameters
            self.iohandler.export_parameters(parameter_list, order, fit_type, captype)

            if fitterconstants.FULL_FIT:
                self.iohandler.generate_Netlist_2_port_full_fit(parameter_list[0],order, fit_type, saturation_table, captype=captype)
            else:
                self.iohandler.generate_Netlist_2_port(parameter_list[0],order, fit_type, saturation_table)

            for it, fitter in enumerate(fitters):
                upper_frq_lim = fitterconstants.FREQ_UPPER_LIMIT

                fitter.write_model_data(parameter_list[it], order)

                self.iohandler.output_plot(
                    fitter.frequency_vector[fitter.frequency_vector < upper_frq_lim],
                    fitter.z21_data[fitter.frequency_vector < upper_frq_lim],
                    fitter.data_mag[fitter.frequency_vector < upper_frq_lim],
                    fitter.data_ang[fitter.frequency_vector < upper_frq_lim],
                    fitter.model_data[fitter.frequency_vector < upper_frq_lim],
                    fitter.file.name)



            ################ END OUTPUT ################################################################################

        except Exception as e:
            self.logger.error("ERROR: An Exception occurred during execution:")
            self.logger.error(str(e) + '\n')

        finally:
            plt.show()
            self.fitter = None

        return 0

    ####################################################################################################################
    # auxilliary functions

    def match_parameters(self, parameter_list, fitters, captype = None):

        orders = [fitter.order for fitter in fitters]

        w_array = np.full(( len(parameter_list), max(orders)), None)

        for num_set, parameter_set in enumerate(parameter_list[:]):
            for key_number in range(1, orders[num_set] + 1):
                w_key = "w%s" % key_number
                w_array[num_set, key_number-1] = parameter_set[w_key].value


        #create an assignment matrix, looking for the resonance in the next dataset (relative keys)
        assignment_matrix = np.full(( len(parameter_list), max(orders)), None)

        for set_number in range(1, np.shape(w_array)[0]):
            ref_array = list(filter(lambda x: x is not None, w_array[set_number - 1]))
            for param_number in range(np.shape(w_array)[1]):
                if not w_array[set_number][param_number] is None:
                    diff = abs(ref_array - w_array[set_number][param_number])
                    best_match = np.argwhere(diff == min(diff))[0][0]
                    assignment_matrix[set_number, param_number] = best_match

        #rebuild the matrix to have absolute keys rather than relative ones

        asg_mat_new = np.full(( len(parameter_list), max(orders)), None)

        for set_number in reversed(range(1, np.shape(w_array)[0])):
            for param_number in range(np.shape(w_array)[1]):
                if not assignment_matrix[set_number][param_number] is None:
                    rel_key = assignment_matrix[set_number][param_number]
                    for backwards_set_number in reversed(range(1, set_number)):
                        rel_key = assignment_matrix[backwards_set_number][rel_key]
                    abs_key = rel_key
                    asg_mat_new[set_number][param_number] = abs_key


        assignment_matrix = asg_mat_new

        match fitters[0].fit_type: #TODO: this could use some better way of determining the fit type
            case fitterconstants.El.INDUCTOR:
                r_default = 1e-1
            case fitterconstants.El.CAPACITOR:
                r_default = 1e9

        #switch key numbers
        for set_number in range(1, np.shape(w_array)[0]):
            parameter_set = parameter_list[set_number]
            previous_set  = parameter_list[set_number - 1]

            output_set = Parameters()
            output_set = self.copy_nominals(output_set, parameter_set, fitters[0].fit_type, captype)

            for param_number in range(np.shape(w_array)[1]):
                old_key_nr = param_number + 1
                if assignment_matrix[set_number][param_number] is not None:
                    new_key_nr = assignment_matrix[set_number][param_number] + 1

                if assignment_matrix[set_number][param_number] is not None:
                    output_set = self.switch_key(output_set, parameter_set, old_key_nr, new_key_nr)

            #fill remaining keys
            for check_key in range(1, np.shape(w_array)[1] + 1):
                w_key = "w%s" % check_key
                if not w_key in output_set:
                    output_set = self.fill_key(output_set, previous_set, check_key, r_default)

            parameter_list[set_number] = copy.copy(output_set)

        return parameter_list

    def copy_nominals(self,out_set, parameter_set, fit_type, captype = None):
        match fit_type:
            case fitterconstants.El.INDUCTOR:
                out_set.add('R_s', value = parameter_set['R_s'].value)
                out_set.add('R_Fe', value =parameter_set['R_Fe'].value)
                out_set.add('L', value =parameter_set['L'].value)
                out_set.add('C', value =parameter_set['C'].value)
            case fitterconstants.El.CAPACITOR:
                out_set.add('R_s', value =parameter_set['R_s'].value)
                out_set.add('R_iso', value =parameter_set['R_iso'].value)
                out_set.add('L', value =parameter_set['L'].value)
                out_set.add('C', value =parameter_set['C'].value)
                if captype == fitterconstants.captype.MLCC:
                    out_set.add('R_A', value=parameter_set['R_A'].value)
                    out_set.add('L_A', value=parameter_set['L_A'].value)
                    out_set.add('C_A', value=parameter_set['C_A'].value)

        return out_set

    def fill_key(self, parameter_set, previous_param_set, key_to_fill, r_value):
        w_key  = "w%s"  % key_to_fill
        BW_key = "BW%s" % key_to_fill
        R_key  = "R%s"  % key_to_fill
        L_key  = "L%s"  % key_to_fill
        C_key  = "C%s"  % key_to_fill

        previous_param_set[R_key].expr = ''
        previous_param_set[L_key].expr = ''
        previous_param_set[C_key].expr = ''

        parameter_set.add(w_key, value=previous_param_set[w_key].value)
        parameter_set.add(BW_key,value=previous_param_set[BW_key].value)
        parameter_set.add(R_key, value=r_value)
        parameter_set.add(L_key, value=previous_param_set[L_key].value)
        parameter_set.add(C_key, value=previous_param_set[C_key].value)

        return parameter_set

    def switch_key(self, parameter_set_out, parameter_set_in, old_key_number, new_key_number):
        old_w_key  = "w%s"  % old_key_number
        old_BW_key = "BW%s" % old_key_number
        old_R_key  = "R%s"  % old_key_number
        old_L_key  = "L%s"  % old_key_number
        old_C_key  = "C%s"  % old_key_number

        new_w_key  = "w%s"  % new_key_number
        new_BW_key = "BW%s" % new_key_number
        new_R_key  = "R%s"  % new_key_number
        new_L_key  = "L%s"  % new_key_number
        new_C_key  = "C%s"  % new_key_number

        parameter_set_in[old_R_key].expr = ''
        parameter_set_in[old_L_key].expr = ''
        parameter_set_in[old_C_key].expr = ''

        parameter_set_out.add(new_w_key, value = parameter_set_in[old_w_key].value)
        parameter_set_out.add(new_BW_key, value = parameter_set_in[old_BW_key].value)
        parameter_set_out.add(new_R_key, value = parameter_set_in[old_R_key].value)
        parameter_set_out.add(new_L_key, value = parameter_set_in[old_L_key].value)
        parameter_set_out.add(new_C_key, value = parameter_set_in[old_C_key].value)

        return parameter_set_out

    def generate_saturation_table(self, parameter_list, key, dc_bias_values):
        #aux function to create saturation tables; since we need a lot of those, especially when doing full fit, it
        #might prove useful and helps keep the code tidy

        #initialize an empty string
        saturation_table = ''

        #check if we have the requested parameter -> else write the default sat table (0,1) i.e. no change with DC bias
        try:
            nominal = parameter_list[0][key].value
        except:
            print('Parameter ' + key + ' does not exist, can\'t create saturation table')
            saturation_table = '0.0,1.0'
            return saturation_table

        for i, value in enumerate(dc_bias_values):
            try:
                if key in parameter_list[i]:
                    saturation_table += str(value) + ','
                    saturation_table += str(parameter_list[i][key].value / nominal)

                if value != dc_bias_values[-1] and key in parameter_list[i + 1]:
                    saturation_table += ','
            except:
                pass

        return saturation_table


    # check function, invoked by the validationcommand of the entry-fields
    # returns TRUE if the value entried is a number; can have a leading + or - and can have ONE decimal point (.)
    # returns FALSE otherwise
    def entry_number_callback(self, checkstring):
        # regular expression copied from: https://stackoverflow.com/questions/46116037/tkinter-restrict-entry-data-to-float
        regex = re.compile(r"(\+|\-)?[0-9.]*$")

        # https://regexlib.com/REDetails.aspx?regexp_id=857
        # regex = re.compile("\b-?[1-9](?:\.\d+)?[Ee][-+]?\d+\b")

        result = regex.match(checkstring)

        checkval = (checkstring == ""
                    or (checkstring.count('+') <= 1
                        and checkstring.count('-') <= 1
                        and checkstring.count('.') <= 1
                        and result is not None
                        and result.group(0) != ""))
        # print(checkval)
        return checkval

    #function to cast the strings from the entry boxes to float, if it does not work, "None" is returned
    #TODO: look into proper error handling here
    def entry_to_float (self, number_string):
        try:
            return float(number_string)
        except:
            return None


    def start(self):
        self.root.mainloop()
