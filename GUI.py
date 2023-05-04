# import packages
import tkinter as tk
from tkinter import filedialog

import constants


from fitter import *
from iohandler import *
from cmc_fitter import *
import GUI_config
import constants
import config
import copy
import os
import re
import skrf as rf
from tkinter import scrolledtext
from texthandler import *
from lmfit import Parameters

import multiprocessing as mp

class GUI:
    """
    The GUI is responsible for creating the user interface and dispatching tasks to the other classes, it instantiates
    most things and is essentially the "heart" of this program
    """
    def __init__(self):
        # declare instance variables
        self.st = None
        self.texthndl = None
        self.entry_saturation = None
        self.entry_nominal_value = None
        self.entry_resistance = None
        self.entry_prominence = None
        self.browse_button = None
        self.shunt_series = None
        self.selected_s2p_files = None

        self.iohandler = None
        self.fitter = None
        self.logger = None
        # variables for the file list
        self.filelist_frame = None
        self.filename_label = []
        self.filename_entry = []
        self.filename_ref_button = []
        self.ref_file_select =None
        #variables for cmcs
        self.cmc_files = {}
        self.checklables = {}

        # Window GUI_config
        self.root: tk.Tk = tk.Tk()
        self.root.wm_title('ATMIS')
        self.root.config(bg='#FFFFFF')

        # # print screen size
        # print("Width: ", self.root.winfo_screenwidth())
        # print("Height: ", self.root.winfo_screenheight())

        # set window size
        self.root.geometry(
            "%dx%d" % (int(GUI_config.GUI_REL_WIDTH * self.root.winfo_screenwidth()),
                       int(GUI_config.GUI_REL_HEIGHT * self.root.winfo_screenheight())))

        # here starts the creation of the widgets
        self.create_drop_down()
        self.create_specification_field()
        self.create_browse_button()
        self.create_run_button()
        self.create_log_window()
        self.create_shunt_series_radio_button()
        self.create_filelist_frame()
        self.create_clear_files_button()

        IOhandleinstance = IOhandler(self.logger)
        self.iohandler = IOhandleinstance

        self.root.mainloop()

    def create_drop_down(self):
        """
        Method to create the dropdown menu for DUT type selection

        :return: None
        """
        self.drop_down_var = tk.StringVar(self.root, GUI_config.DROP_DOWN_ELEMENTS[0])
        self.drop_down_var.trace_add('write', self.drop_down_update_callback)

        self.option_menu = tk.OptionMenu(self.root, self.drop_down_var, *GUI_config.DROP_DOWN_ELEMENTS)
        max_drop_length = len(max(GUI_config.DROP_DOWN_ELEMENTS, key=len))
        self.option_menu.config(font=GUI_config.DROP_DOWN_FONT, width=max_drop_length + 5, height=GUI_config.DROP_DOWN_HEIGHT)
        self.option_menu.grid(column=0, row=0, columnspan=1, sticky=tk.W, **GUI_config.HEADLINE_PADDING)

    def drop_down_update_callback(self, var, index, mode):
        """
        method to set the filelist frame to CMC config or coil/cap config

        :param var:
        :param index:
        :param mode:
        :return:
        """
        selected_element = self.drop_down_var.get()
        if selected_element == GUI_config.DROP_DOWN_ELEMENTS[2]:
            #change GUI to CMC
            self.filelist_frame.destroy()
            self.browse_button.destroy()
            self.callback_clear_files()
            self.create_cmc_frame()
            self.iohandler.files = []
        else:
            if selected_element == GUI_config.DROP_DOWN_ELEMENTS[1]: #capacitor
                self.create_captype_dropdown()
            elif selected_element == GUI_config.DROP_DOWN_ELEMENTS[0]: #inductor
                self.captype_menu.destroy()
            #rebuild GUI for coil/cap config
            self.filelist_frame.destroy()
            self.create_browse_button()
            self.create_filelist_frame()
            self.callback_clear_files() #TODO: idk if we should remove all files when selecting a different element
            self.cmc_files = {}
            self.iohandler.files = []

    def create_captype_dropdown(self):
        """
        function to create a dropdown menu for the various types of capacitors
        :return:
        """
        self.captype_var = tk.StringVar(self.root, GUI_config.CAPTYPE_DROPDOWN_ELEMENTS[0])

        self.captype_menu = tk.OptionMenu(self.root, self.captype_var, *GUI_config.CAPTYPE_DROPDOWN_ELEMENTS)
        max_drop_length = len(max(GUI_config.CAPTYPE_DROPDOWN_ELEMENTS, key=len))
        self.captype_menu.config(font=GUI_config.DROP_DOWN_FONT, width=20,
                                height=GUI_config.DROP_DOWN_HEIGHT,)
        self.captype_menu.grid(column=1, row=0, columnspan=1, sticky=tk.W, **GUI_config.HEADLINE_PADDING)

    def return_captype(self):
        """
        function to return the captype from the drop down menu to the "run" function
        :return: captype; integer variable representing the type of capacitor
        """
        captype_string =  self.captype_var.get()

        if captype_string == GUI_config.CAPTYPE_DROPDOWN_ELEMENTS[0]:
            return constants.captype.GENERIC
        elif captype_string == GUI_config.CAPTYPE_DROPDOWN_ELEMENTS[1]:
            return constants.captype.MLCC
        elif captype_string == GUI_config.CAPTYPE_DROPDOWN_ELEMENTS[2]:
            return constants.captype.HIGH_C

    def create_cmc_frame(self):
        self.filelist_frame = tk.LabelFrame(self.root, text='Files')
        self.filelist_frame.grid(column=4, row=1, rowspan=10, columnspan=2, sticky=tk.NW, **GUI_config.SPEC_PADDING)
        # create headings for the columns
        ref_lbl = tk.Label(self.filelist_frame, text='Mode')
        name_lbl = tk.Label(self.filelist_frame, text='Present')

        ref_lbl.grid(column=0,row=0, sticky="w")
        name_lbl.grid(column=1,row=0)

        #create mode lables
        cmlabel = tk.Label(self.filelist_frame, text='Common Mode')
        dmlabel = tk.Label(self.filelist_frame, text='Differential Mode')
        oclabel = tk.Label(self.filelist_frame, text='Open Circuit')
        cmlabel.grid(column=0, row=1, sticky="w")
        dmlabel.grid(column=0, row=2, sticky="w")
        oclabel.grid(column=0, row=3, sticky="w")


        #create buttons for file selection
        cmbutton = tk.Button(self.filelist_frame, command=lambda:self.load_cmc_file("CM"), text= "Load")
        dmbutton = tk.Button(self.filelist_frame, command=lambda:self.load_cmc_file("DM"), text= "Load")
        ocbutton = tk.Button(self.filelist_frame, command=lambda:self.load_cmc_file("OC"), text= "Load")
        cmbutton.grid(column=2, row=1)
        dmbutton.grid(column=2, row=2)
        ocbutton.grid(column=2, row=3)

        #create checkmark lables
        dmchecklabel = tk.Label(self.filelist_frame, text='')
        cmchecklabel = tk.Label(self.filelist_frame, text='')
        occhecklabel = tk.Label(self.filelist_frame, text='')
        cmchecklabel.grid(column=1, row=1)
        dmchecklabel.grid(column=1, row=2)
        occhecklabel.grid(column=1, row=3)
        self.checklables["DM"] = dmchecklabel
        self.checklables["CM"] = cmchecklabel
        self.checklables["OC"] = occhecklabel

    def load_cmc_file(self, mode: str):
        try:
            filename = tk.filedialog.askopenfilename(title=
                                                     'Load ' + mode + ' Measurement',
                                                     filetypes=((".s2p", "*.s2p*"),
                                                                ("all files", "*.*")), multiple=False)
            #we need to check if the filename is not an empty string (i.e. if the user has not canceled the load)
            if filename:
                ntwk = rf.Network(os.path.abspath(filename))
                self.cmc_files[mode] = ntwk
                self.checklables[mode].config(text = "\u2713")
        except Exception as e:
            raise e

    def create_shunt_series_radio_button(self):
        """
        Method to create the radiobutton for shunt/series through calculation selection

        :return: None
        """
        self.shunt_series = tk.IntVar()
        label_calc = tk.Label(self.root, text="Z Calculation Method")
        label_calc.config(font=GUI_config.ENTRY_FONT)
        label_calc.grid(column=2, row=1, sticky=tk.NW, **GUI_config.HEADLINE_PADDING)

        r1 = tk.Radiobutton(self.root, text = 'Shunt Through', variable=self.shunt_series, value=GUI_config.SHUNT_THROUGH)
        r2 = tk.Radiobutton(self.root, text = 'Series Through', variable=self.shunt_series, value=GUI_config.SERIES_THROUGH)
        r1.grid(column=2, row=2)
        r2.grid(column=2, row=3)

    def create_specification_field(self):
        """
        Method to create the specification fields

        :return: None
        """

        # validate command for inputs "register" is necessary so that the actual input is checked (would otherwise
        # update after input)
        vcmd = (self.root.register(self.entry_number_callback), "%P")

        # Headline
        label_spec = tk.Label(self.root, text="Specification", bg=GUI_config.BCKGND_COLOR)
        label_spec.config(font=GUI_config.HEADLINE_FONT)
        label_spec.grid(column=0, row=1, columnspan=2, sticky=tk.NW, **GUI_config.HEADLINE_PADDING)

        # initial value
        passive_element_label = tk.Label(self.root, text="Nominal value", bg=GUI_config.BCKGND_COLOR)
        passive_element_label.config(font=GUI_config.ENTRY_FONT)
        passive_element_label.grid(column=0, row=2, sticky=tk.W, **GUI_config.SPEC_PADDING)

        self.entry_nominal_value = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_nominal_value.config(font=GUI_config.ENTRY_FONT)
        self.entry_nominal_value.grid(column=1, row=2, sticky=tk.W, **GUI_config.ENTRY_PADDING)

        # initial resistance value
        label_resistance = tk.Label(self.root, text="\u03A9", bg=GUI_config.BCKGND_COLOR)
        label_resistance.config(font=GUI_config.ENTRY_FONT)
        label_resistance.grid(column=0, row=3, sticky=tk.W, **GUI_config.SPEC_PADDING)

        self.entry_resistance = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_resistance.config(font=GUI_config.ENTRY_FONT)
        self.entry_resistance.grid(column=1, row=3, sticky=tk.W, **GUI_config.ENTRY_PADDING)

        # Prominence
        label_prominence = tk.Label(self.root, text="Prominence in dB", bg=GUI_config.BCKGND_COLOR)
        label_prominence.config(font=GUI_config.ENTRY_FONT)
        label_prominence.grid(column=0, row=5, sticky=tk.W, **GUI_config.SPEC_PADDING)

        self.entry_prominence = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_prominence.config(font=GUI_config.ENTRY_FONT)
        self.entry_prominence.grid(column=1, row=5, sticky=tk.W, **GUI_config.ENTRY_PADDING)

    def create_browse_button(self):
        """
        Method to create the "browse files" button

        :return: None
        """
        self.browse_button = tk.Button(self.root, command=self.callback_browse_s2p_file, text="Select s2p File(s)")
        self.browse_button.config(font=GUI_config.ENTRY_FONT)
        self.browse_button.grid(column=0, row=6, sticky=tk.W, **GUI_config.BUTTON_LEFT_PADDING)

    def create_run_button(self):
        """
        Method to create the "run" button

        :return: None
        """
        browse_button = tk.Button(self.root, command=self.callback_run, text="Run")
        browse_button.config(font=GUI_config.ENTRY_FONT)
        browse_button.grid(column=1, row=6, sticky=tk.W, **GUI_config.BUTTON_RIGHT_PADDING)

    def create_clear_files_button(self):
        """
        Method to create the "clear files" button

        :return: None
        """
        browse_button = tk.Button(self.root, command=self.callback_clear_files, text="Clear Files")
        browse_button.config(font=GUI_config.ENTRY_FONT)
        browse_button.grid(column=4, row=0, sticky=tk.W, **GUI_config.BUTTON_RIGHT_PADDING)

    def create_log_window(self):
        """
        Method to create the log window

        :return: None
        """
        self.st = scrolledtext.ScrolledText(self.root, state='disabled')#, width=GUI_config.LOG_WIDTH,  height=GUI_config.LOG_HEIGHT)
        self.st.configure(font='TkFixedFont')
        self.st.grid(column=0, row=9, columnspan=3, sticky=tk.W, **GUI_config.ENTRY_PADDING)
        # self.st.pack()
        self.texthndl = Text_Handler(self.st)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.texthndl)
        self.logger.setLevel(logging.INFO)

    def create_filelist_frame (self):
        """
        Method to create the frame for the file list

        :return: None
        """
        self.filelist_frame = tk.LabelFrame(self.root, text = 'Files')
        self.filelist_frame.grid(column = 4, row = 1, rowspan = 10, columnspan=2, sticky=tk.NW, **GUI_config.SPEC_PADDING)
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
        """
        Method to get the current/voltage values from the entry boxes of the file list

        :return: current_voltage list. List type object containing the DC bias values
        """

        file_current_voltage_list = []

        for entry in self.filename_entry:
            file_current_voltage_list.append(self.entry_to_float(entry.get()))

        return file_current_voltage_list

    def update_file_list(self):
        """
        Method to update the file list after new files have been loaded or files have been cleared

        :return: None
        """
        rownumber = len(self.filename_label) + 1
        vcmd = (self.root.register(self.entry_number_callback), "%P")
        existing_lables = []

        for file in self.iohandler.files:

            for testlabel in self.filename_label:
                existing_lables.append(testlabel.cget("text"))

            if file.name in existing_lables:
                continue

            #create a label for the filename
            label_name = file.name
            label = tk.Label(self.filelist_frame, text= label_name)
            entry = tk.Entry(self.filelist_frame, width = 5, validate='all', validatecommand=(vcmd))
            label.grid(column=1, row = rownumber, sticky=tk.NW, **GUI_config.SPEC_PADDING)
            entry.grid(column=2, row = rownumber, sticky=tk.NSEW, **GUI_config.SPEC_PADDING)
            #create a button for the selection of the reference file
            r_button = tk.Radiobutton(self.filelist_frame, variable=self.ref_file_select, value =rownumber - 1)
            r_button.grid(column=0, row=rownumber)

            rownumber += 1
            self.filename_entry.append(entry)
            self.filename_label.append(label)
            self.filename_ref_button.append(r_button)

    def callback_clear_files(self):
        """
        Callback function of the "clear files" button

        Clears the files present in the IOhandler and destroys the corresponding entry boxes in the file list

        :return: None
        """
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

    def callback_browse_s2p_file(self):
        """
        Callback function for the "browse files" button

        Opens a file dialogue and loads the selected files to the IOhandler. Updates the file list afterwards

        :return: None
        """
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
            self.iohandler.load_file(path_list)
        except Exception as e:
            self.logger.error("ERROR: There was an error, opening one of the selected files:")
            self.logger.error(str(e))

        #insert the files to the listbox
        self.update_file_list()

    def callback_run(self):

        """
        Callback function for the "run" button

        **Calls the corresponding fitting routines**

        :return: None
        """

        if self.drop_down_var.get() == GUI_config.DROP_DOWN_ELEMENTS[2]: #CMC
            self.fit_cmc()
        elif self.drop_down_var.get() == GUI_config.DROP_DOWN_ELEMENTS[1]:#CAP
            self.fit_cap()
        elif self.drop_down_var.get() == GUI_config.DROP_DOWN_ELEMENTS[0]:#COIL
            self.fit_coil()

    def fit_cmc(self):

        Z0 = 50

        #instanciate cmc fitter with logger instance
        fitter_instance = CMC_Fitter(self.logger)

        #iterate through loaded files and safe the keys
        checkkey = []
        for key in self.cmc_files:
            checkkey.append(key)

        #check if all required configurations for CMCs are present
        if not(set(config.CMC_REQUIRED_CONFIGURATIONS).issubset(set(checkkey))):
            raise Exception("not all required files present")




        fitter_instance.set_file(self.cmc_files['CM'])
        fitter_instance.cmcmodel = cmctype.PLATEAU




        fitter_instance.calc_series_thru(Z0)
        fitter_instance.smooth_data()
        fitter_instance.get_main_resonance()
        fitter_instance.calculate_nominal_value()
        fitter_instance.get_resonances()

        fitter_instance.create_nominal_parameters_CM()
        fitter_instance.plot_curves_cmc(1)
        #for testing:
        fitter_instance.params_dict["DM"].pretty_print()
        fitter_instance.fit_cmc_main_res()
        fitter_instance.plot_curves_cmc(1)
        fitter_instance.params_dict["DM"].pretty_print()
        fitter_instance.create_higher_order_parameters()
        fitter_instance.plot_curves_cmc(0)
        fitter_instance.fit_cmc_higher_order_res()
        fitter_instance.plot_curves_cmc(0)

        pass
        # fitter_instance.one_sided_params_to_sym_params()



        pass

    def fit_coil(self):

        fit_type = constants.El.INDUCTOR


        self.logger.info("----------Run----------\n")
        [passive_nom, res, prom, shunt_series, files, dc_bias] = self.read_from_GUI()

        try:

            ################ PARSING AND PRE-PROCESSING ################################################################

            #create an array for the fitter instances and one for the parameters
            fitters = []
            parameter_list = []

            for it, file in enumerate(files):
                #instanciate a fitter and pass it the file and the logger instance
                fitter_instance = Fitter(self.logger)
                fitter_instance.set_file(file)

                #calculate the impedance data for the fitter
                match shunt_series:
                    case GUI_config.SHUNT_THROUGH:
                        fitter_instance.calc_shunt_thru(GUI_config.Z0)
                    case GUI_config.SERIES_THROUGH:
                        fitter_instance.calc_series_thru(GUI_config.Z0)

                #smooth the impedance data and pass the specification
                fitter_instance.smooth_data()
                fitter_instance.set_specification(passive_nom, res, prom, fit_type, captype)

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
                     fitted_main_res_params = fitter.fit_main_res_inductor_file_1(main_res_params)
                else:
                    #fit the main resonance for every other file (we have to overwrite some parameters here, since the
                    # main parasitic element (C for inductors, L for capacitors) and the R_s should be constrained
                    main_res_params = fitter.overwrite_main_res_params_file_n(main_res_params, parameter_list[0])
                    fitted_main_res_params = fitter.fit_main_res_inductor_file_n(main_res_params)
                #finally write the fitted main resonance parameters to the list
                parameter_list.append(fitted_main_res_params)

            ################ END MAIN RESONANCE FIT ####################################################################

            '''
            ################ HIGHER ORDER RESONANCES - SINGLE PROCESS ##################################################

            test_pre_fit = []
            for it, fitter in enumerate(fitters):

                fitter.get_resonances()

                if fitter.order:
                    #generate parameter sets in two configurations 1=Q constrained; 2=free R and C
                    params1 = copy.copy(fitter.create_higher_order_parameters(1, parameter_list[it]))
                    params2 = copy.copy(fitter.create_higher_order_parameters(2, parameter_list[it]))

                    #correct obtained parameters
                    correct_main_res = False
                    num_iterations = 4

                    params1 = fitter.correct_parameters(params1, correct_main_res, num_iterations)
                    params2 = fitter.correct_parameters(params2, correct_main_res, num_iterations)


                    params1 = fitter.pre_fit_bands(params1)
                    params2 = fitter.pre_fit_bands(params2)

                    #TODO: delme
                    test_pre_fit.append(params1)

                    #fit the whole curve
                    fit_params1 = fitter.fit_curve_higher_order(params1)
                    fit_params2 = fitter.fit_curve_higher_order(params2)

                    #check wich model fits best
                    out_params = fitter.select_param_set([fit_params1, fit_params2], debug=True)
                    parameter_list[it] = out_params

                    #write the model data to the fitter instance
                    # fitter.write_model_data(out_params)

                    if DEBUG_FIT:
                        fitter.plot_curve(parameter_list[it], fitter.order, False, str(fitter.file.name) + ' fitted higher order resonances')


            ################ END HIGHER ORDER RESONANCES - SINGLE PROCESS ##############################################
            '''

            ################ HIGHER ORDER RESONANCES - MULTIPROCESSING POOL ############################################
            # mp.freeze_support()
            self.mp_pool = mp.Pool(config.MULTIPROCESSING_COUNT)


            for it, fitter in enumerate(fitters):
                fitter.get_resonances()

            correct_main_res = False
            num_iterations = 4
            temp_param_list = []
            for it, fitter in enumerate(fitters):
                params1 = copy.copy(fitter.create_higher_order_parameters(2, parameter_list[it]))
                temp_param_list.append(fitter.correct_parameters(params1, correct_main_res, num_iterations))

            #apply pre-fitting tasks to the multiprocessing pool
            pre_fit_results = []
            for it, fitter in enumerate(fitters):
                pre_fit_results.append(self.mp_pool.apply_async(fitter.pre_fit_bands, args = (temp_param_list[it],)))

            #wait for all pre-fits to finish
            [result.wait() for result in pre_fit_results]

            #write back to parameter list
            for it, pre_fit_result in enumerate(pre_fit_results):
                temp_param_list[it] = copy.copy(pre_fit_result.get())

            #CURVE FIT
            #create array for fit results
            fit_results = []
            for it, fitter in enumerate(fitters):
                fit_results.append(self.mp_pool.apply_async(fitter.fit_curve_higher_order, args = (temp_param_list[it],)))

            # wait for all pre-fits to finish
            [result.wait() for result in fit_results]

            # write back to parameter list
            for it, result in enumerate(fit_results):
                parameter_list[it] = copy.copy(result.get())

            for it, fitter in enumerate(fitters):
                if DEBUG_FIT:
                    fitter.plot_curve(parameter_list[it], fitter.order, False, str(fitter.file.name) + ' fitted higher order resonances')

            #fit done
            self.mp_pool.close()

            ################ END HIGHER ORDER RESONANCES - MULTIPROCESSING POOL ########################################


            ############### MATCH PARAMETERS ###########################################################################

            parameter_list = self.match_parameters(parameter_list, fitters, captype)

            ############### END MATCH PARAMETERS #######################################################################

            ################ SATURATION TABLE(S) #######################################################################

            order = max([fitter.order for fitter in fitters])
            # saturation table for nominal value
            # create saturation table and get nominal value
            saturation_table = {}
            match fit_type:
                case constants.El.INDUCTOR:
                    saturation_table['L'] = self.generate_saturation_table(parameter_list, 'L', dc_bias)
                    saturation_table['R_Fe'] = self.generate_saturation_table(parameter_list, 'R_Fe',
                                                                              dc_bias)
                case constants.El.CAPACITOR:
                    saturation_table['C'] = self.generate_saturation_table(parameter_list, 'C', dc_bias)
                    saturation_table['R_s'] = self.generate_saturation_table(parameter_list, 'R_s', dc_bias)

            # write saturation table for acoustic resonance
            if fit_type == constants.El.CAPACITOR and captype == constants.captype.MLCC:
                saturation_table['R_A'] = self.generate_saturation_table(parameter_list, 'R_A', dc_bias)
                saturation_table['L_A'] = self.generate_saturation_table(parameter_list, 'L_A', dc_bias)
                saturation_table['C_A'] = self.generate_saturation_table(parameter_list, 'C_A', dc_bias)

            if config.FULL_FIT:

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

            if config.FULL_FIT:
                self.iohandler.generate_Netlist_2_port_full_fit(parameter_list[0],order, fit_type, saturation_table, captype=captype)
            else:
                self.iohandler.generate_Netlist_2_port(parameter_list[0],order, fit_type, saturation_table, captype=captype)

            for it, fitter in enumerate(fitters):
                upper_frq_lim = config.FREQ_UPPER_LIMIT

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

    def fit_cap(self):

        # TODO: consider some entry box or something
        captype = self.return_captype()

        # fit type is capacitor for this method
        fit_type = constants.El.CAPACITOR

        self.logger.info("----------Run----------\n")

        [passive_nom, res, prom, shunt_series, files, dc_bias] = self.read_from_GUI(captype)

        #set prominence to 3dB in case of High C model, because we need to avoid misdetection of resonances here
        if captype == constants.captype.HIGH_C:
            prom = 3

        try:
            ################ PARSING AND PRE-PROCESSING ################################################################

            #create an array for the fitter instances and one for the parameters
            fitters = []
            parameter_list = []

            for it, file in enumerate(files):
                #instanciate a fitter and pass it the file and the logger instance
                fitter_instance = Fitter(self.logger)
                fitter_instance.set_file(file)

                #calculate the impedance data for the fitter
                match shunt_series:
                    case GUI_config.SHUNT_THROUGH:
                        fitter_instance.calc_shunt_thru(GUI_config.Z0)
                    case GUI_config.SERIES_THROUGH:
                        fitter_instance.calc_series_thru(GUI_config.Z0)

                #smooth the impedance data and pass the specification
                fitter_instance.smooth_data()
                fitter_instance.set_specification(passive_nom, res, prom, fit_type, captype)

                #write instance to list
                fitters.append(fitter_instance)

            ################ END PARSING AND PRE-PROCESSING ############################################################

            ################ HIGH C RESONANCE DETECTION ################################################################
            if captype == constants.captype.HIGH_C:
                #we need to specify some resonance frequency even if there is no detectable resonant frequency
                # yet the f0 is required for some routines, hence we set it to an arbitrary value lower than the first resonance
                for it, fitter in enumerate(fitters):
                    fitter.f0 = 0
                    freq = fitter.frequency_vector
                    fitter.get_resonances()
                    try:
                        lowest_res = fitter.bandwidths[0][1]
                        fitter.f0 = lowest_res / 8
                    except:
                        # if we didn't find anything, we just set it to 10e5... why not I guess
                        fitter.f0 = 10e5

                    #TODO: this can lead to trouble if the resonant frequency is set to such a low value that there are
                    # no values at frequencies lower than f0
                    #also we need to account for R_s
                    R_s = abs(np.mean(fitter.z21_data[freq < fitter.f0]))
                    fitter.series_resistance = R_s


            ################ END HIGH C RESONANCE DETECTION ############################################################

            ################ HIGH C MODEL ##############################################################################
            if captype == constants.captype.HIGH_C:

                for it, fitter in enumerate(fitters):
                    params = Parameters()
                    main_res_params = fitter.create_hi_C_parameters(params)
                    fitted_model_params = fitter.fit_hi_C_model(main_res_params)
                    parameter_list.append(fitted_model_params)

                # angleplot = False
                #
                # fitter.plot_curve(main_res_params, 0, 1, "onlyMR nonfit", angle = angleplot)
                #
                # fitter.plot_curve(fitted_model_params, 0, 1,"onlyMR fit", angle = angleplot)
                # if fitter.order:
                #     fitter.plot_curve(fitted_model_params, 1, 0,"fit", angle = angleplot)
                #     fitter.plot_curve(main_res_params, 1, 0, "nonfit", angle = angleplot)



            ################ END HIGH C MODEL ##########################################################################

            ################ MAIN RESONANCE FIT ########################################################################
            if captype != constants.captype.HIGH_C:
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
                        fitted_main_res_params = fitter.fit_main_res_capacitor_file_1(main_res_params)
                    else:
                        #fit the main resonance for every other file (we have to overwrite some parameters here, since the
                        # main parasitic element (C for inductors, L for capacitors) and the R_s should be constrained
                        main_res_params = fitter.overwrite_main_res_params_file_n(main_res_params, parameter_list[0])
                        fitted_main_res_params = fitter.fit_main_res_capacitor_file_n(main_res_params)

                    #finally write the fitted main resonance parameters to the list
                    parameter_list.append(fitted_main_res_params)

                ################ END MAIN RESONANCE FIT ####################################################################

                ################ ACOUSITC RESONANCE DETECTION FOR MLCCs ####################################################

                # get acoustic resonance frequency for all files, if not found write "None" to list
                if captype == constants.captype.MLCC and fit_type == constants.El.CAPACITOR and len(fitters) > 1:
                    acoustic_res_frqs = []
                    #append None for first file
                    acoustic_res_frqs.append(None)
                    for fitter in fitters[1:]:
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
                        captype = constants.captype.GENERIC
                        for fitter in fitters:
                            fitter.set_captype(captype)

                ################ END ACOUSITC RESONANCE DETECTION FOR MLCCs ################################################
                '''
                ################ HIGHER ORDER RESONANCES - SINGLE PROCESS ##################################################
    
                test_pre_fit = []
                for it, fitter in enumerate(fitters):
    
                    fitter.get_resonances()
    
                    if fitter.order:
                        #generate parameter sets in two configurations 1=Q constrained; 2=free R and C
                        params1 = copy.copy(fitter.create_higher_order_parameters(1, parameter_list[it]))
                        params2 = copy.copy(fitter.create_higher_order_parameters(2, parameter_list[it]))
    
                        #correct obtained parameters
                        correct_main_res = False
                        num_iterations = 4
    
                        params1 = fitter.correct_parameters(params1, correct_main_res, num_iterations)
                        params2 = fitter.correct_parameters(params2, correct_main_res, num_iterations)
    
    
                        params1 = fitter.pre_fit_bands(params1)
                        params2 = fitter.pre_fit_bands(params2)
    
                        #TODO: delme
                        test_pre_fit.append(params1)
    
                        #fit the whole curve
                        fit_params1 = fitter.fit_curve_higher_order(params1)
                        fit_params2 = fitter.fit_curve_higher_order(params2)
    
                        #check wich model fits best
                        out_params = fitter.select_param_set([fit_params1, fit_params2], debug=True)
                        parameter_list[it] = out_params
    
                        #write the model data to the fitter instance
                        # fitter.write_model_data(out_params)
    
                        if DEBUG_FIT:
                            fitter.plot_curve(parameter_list[it], fitter.order, False, str(fitter.file.name) + ' fitted higher order resonances')
    
    
                ################ END HIGHER ORDER RESONANCES - SINGLE PROCESS ##############################################
                '''

                ################ HIGHER ORDER RESONANCES - MULTIPROCESSING POOL ############################################
                self.mp_pool = mp.Pool(config.MULTIPROCESSING_COUNT)

                for it, fitter in enumerate(fitters):
                    fitter.get_resonances()

                correct_main_res = False
                num_iterations = 4
                temp_param_list = []
                for it, fitter in enumerate(fitters):
                    params1 = copy.copy(fitter.create_higher_order_parameters(2, parameter_list[it]))
                    temp_param_list.append(fitter.correct_parameters(params1, correct_main_res, num_iterations))

                #apply pre-fitting tasks to the multiprocessing pool
                pre_fit_results = []
                for it, fitter in enumerate(fitters):
                    pre_fit_results.append(self.mp_pool.apply_async(fitter.pre_fit_bands, args = (temp_param_list[it],)))

                #wait for all pre-fits to finish
                [result.wait() for result in pre_fit_results]

                #write back to parameter list
                for it, pre_fit_result in enumerate(pre_fit_results):
                    temp_param_list[it] = copy.copy(pre_fit_result.get())

                #CURVE FIT
                #create array for fit results
                fit_results = []
                for it, fitter in enumerate(fitters):
                    fit_results.append(self.mp_pool.apply_async(fitter.fit_curve_higher_order, args = (temp_param_list[it],)))

                # wait for all pre-fits to finish
                [result.wait() for result in fit_results]

                # write back to parameter list
                for it, result in enumerate(fit_results):
                    parameter_list[it] = copy.copy(result.get())

                for it, fitter in enumerate(fitters):
                    if DEBUG_FIT:
                        fitter.plot_curve(parameter_list[it], fitter.order, False, str(fitter.file.name) + ' fitted higher order resonances')

                #fit done
                self.mp_pool.close()

                ################ END HIGHER ORDER RESONANCES - MULTIPROCESSING POOL ########################################


            ############### MATCH PARAMETERS ###########################################################################

            parameter_list = self.match_parameters(parameter_list, fitters, captype)

            ############### END MATCH PARAMETERS #######################################################################

            ################ SATURATION TABLE(S) #######################################################################

            order = max([fitter.order for fitter in fitters])
            # saturation table for nominal value
            # create saturation table and get nominal value
            saturation_table = {}
            match fit_type:
                case constants.El.INDUCTOR:
                    saturation_table['L'] = self.generate_saturation_table(parameter_list, 'L', dc_bias)
                    saturation_table['R_Fe'] = self.generate_saturation_table(parameter_list, 'R_Fe',
                                                                              dc_bias)
                case constants.El.CAPACITOR:
                    saturation_table['C'] = self.generate_saturation_table(parameter_list, 'C', dc_bias)
                    saturation_table['R_s'] = self.generate_saturation_table(parameter_list, 'R_s', dc_bias)

            # write saturation table for acoustic resonance
            if fit_type == constants.El.CAPACITOR and captype == constants.captype.MLCC:
                saturation_table['R_A'] = self.generate_saturation_table(parameter_list, 'R_A', dc_bias)
                saturation_table['L_A'] = self.generate_saturation_table(parameter_list, 'L_A', dc_bias)
                saturation_table['C_A'] = self.generate_saturation_table(parameter_list, 'C_A', dc_bias)

            if config.FULL_FIT:

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

            if config.FULL_FIT:
                self.iohandler.generate_Netlist_2_port_full_fit(parameter_list[0],order, fit_type, saturation_table, captype=captype)
            else:
                self.iohandler.generate_Netlist_2_port(parameter_list[0],order, fit_type, saturation_table, captype=captype)

            for it, fitter in enumerate(fitters):
                upper_frq_lim = config.FREQ_UPPER_LIMIT

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

    def read_from_GUI(self, captype = None):
        """
        function to read values from the several entry boxes, radiobuttons etc from the GUI
        :return:
        """

        # get values from the entry boxes
        passive_nom = self.entry_to_float(self.entry_nominal_value.get())
        res = self.entry_to_float(self.entry_resistance.get())
        prom = self.entry_to_float(self.entry_prominence.get())

        # get the shunt/series through setting
        shunt_series = self.shunt_series.get()


        try:
            #check if nominal value is present for the hi C model
            if passive_nom is None and captype == constants.captype.HIGH_C:
                raise Exception("Error: Nominal value is required for High C model")
            # raise an exception if shunt/series through was not set
            if not (shunt_series):
                raise Exception("Error: Shunt/Series-Through not set!\nPlease select a calculation mode")
            # check if files are present
            if not self.iohandler.files:
                raise Exception("Error: No Files present")
            # get selected reference file and make a list with all files that are not the reference file
            ref_file = self.iohandler.files[self.ref_file_select.get()]
            other_files = self.iohandler.files[:self.ref_file_select.get()] + self.iohandler.files[self.ref_file_select.get() + 1:]

            if ref_file is None:
                raise Exception("Error: Please select a reference file")

            # get the values from the entries that define the currents/voltages of each file
            dc_bias = self.get_file_current_voltage_values()
            if None in dc_bias:
                raise Exception("Error: Please specify the current/voltage values for the given files!")

            # the reference file has to be the first file in the list, but also the DC bias needs to match, so we'll
            # have to shuffle around a bit
            dc_bias.insert(0, dc_bias.pop(self.ref_file_select.get()))
            files = [ref_file] + other_files


            return [passive_nom, res, prom, shunt_series, files, dc_bias]

        except Exception as e:
            self.logger.error(str(e) + '\n')
            raise


    ####################################################################################################################
    # auxilliary functions

    def match_parameters(self, parameter_list, fitters, captype = None):
        """
        Auxilliary method to map the parameters of the model to their corresponding frequencies

        :param parameter_list: A list containing the Parameters() objects of all files
        :param fitters: A list containing the instances of all fitters used
        :param captype: Type of capacitor can be GENERIC or MLCC
        :return: A list containing the Parameters() of all files, now with each resonance matched to their corresponding
            frequency
        """

        orders = [fitter.order for fitter in fitters]

        w_array = np.full(( len(parameter_list), max(orders)), None)

        for num_set, parameter_set in enumerate(parameter_list[:]):
            for key_number in range(1, orders[num_set] + 1):
                w_key = "w%s" % key_number
                w_array[num_set, key_number-1] = parameter_set[w_key].value


        #create an assignment matrix, looking for the resonance in the next dataset (relative keys)
        assignment_matrix = np.atleast_2d(np.full(( len(parameter_list), max(orders)), None))

        for set_number in range(1, np.shape(w_array)[0]):
            ref_array = list(filter(lambda x: x is not None, w_array[set_number - 1]))
            for param_number in range(np.shape(w_array)[1]):
                if not w_array[set_number][param_number] is None and ref_array:
                    diff = abs(ref_array - w_array[set_number][param_number])
                    best_match = np.argwhere(diff == min(diff))[0][0]
                    assignment_matrix[set_number, param_number] = best_match

        #rebuild the matrix to have absolute keys rather than relative ones

        asg_mat_new = np.full(( len(parameter_list), max(orders)), None)

        for set_number in reversed(range(1, np.shape(w_array)[0])):
            for param_number in range(np.shape(w_array)[1]):
                if not assignment_matrix[set_number][param_number] is None:
                    rel_key = assignment_matrix[set_number][param_number]
                    if not rel_key is None:
                        for backwards_set_number in reversed(range(1, set_number)):
                            if not assignment_matrix[backwards_set_number][rel_key] is None:
                                rel_key = assignment_matrix[backwards_set_number][rel_key]
                        abs_key = rel_key
                        asg_mat_new[set_number][param_number] = abs_key


        assignment_matrix = asg_mat_new

        match fitters[0].fit_type: #TODO: this could use some better way of determining the fit type
            case constants.El.INDUCTOR:
                r_default = config.R_FILL_IND
            case constants.El.CAPACITOR:
                r_default = config.R_FILL_CAP


        #check if first parameters object has all keys needed, else fill first array with all keys
        for check_key in range(1, np.shape(w_array)[1]+1):
            w_key = "w%s" % check_key
            if not w_key in parameter_list[0]:
                first_occurence = np.argwhere(assignment_matrix[:, param_number] != None)[0][0]
                first_occurence_set = parameter_list[first_occurence]
                parameter_list[0] = self.fill_key(parameter_list[0], first_occurence_set, check_key, r_default)

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
        """
        Auxilliary method to copy the parameters of the main element to a new parameter set

        :param out_set: A Parameters() object to be written to
        :param parameter_set: A Parameters() object from which to copy
        :param fit_type: The type of DUT to fit (coil or capacitor)
        :param captype: The type of capacitor. Can be GENERIC or MLCC
        :return: The out_set with copied main element parameters
        """
        match fit_type:
            case constants.El.INDUCTOR:
                out_set.add('R_s', value = parameter_set['R_s'].value)
                out_set.add('R_Fe', value =parameter_set['R_Fe'].value)
                out_set.add('L', value =parameter_set['L'].value)
                out_set.add('C', value =parameter_set['C'].value)
            case constants.El.CAPACITOR:
                out_set.add('R_s', value =parameter_set['R_s'].value)
                out_set.add('R_iso', value =parameter_set['R_iso'].value)
                out_set.add('L', value =parameter_set['L'].value)
                out_set.add('C', value =parameter_set['C'].value)
                if captype == constants.captype.MLCC:
                    out_set.add('R_A', value=parameter_set['R_A'].value)
                    out_set.add('L_A', value=parameter_set['L_A'].value)
                    out_set.add('C_A', value=parameter_set['C_A'].value)

        return out_set

    def fill_key(self, parameter_set, previous_param_set, key_to_fill, r_value):
        """
        Method to fill a key if a resonance is no longer present in the current parameter set but was present in the
        last parameter set

        :param parameter_set: A Parameters() object to which to write
        :param previous_param_set: The Parameters() object of the previous file
        :param key_to_fill: The resonant circuit to fill
        :param r_value: The value which shall be written to the resistor of the circuit
        :return: The parameter_set to which to write
        """
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
        """
        Auxilliary method to switch a parameter key, necessary when a resonant circuit needs to be mapped to a different
            frequency

        :param parameter_set_out: The Parameters() object to which to write
        :param parameter_set_in: The Parameters() set containing the key to map to a different key
        :param old_key_number: The number(key) which needs to be mapped
        :param new_key_number: The number(key) to map to
        :return: The parameter_set_out
        """
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
        """
        Auxilliary function to generate saturation tables.

        Saturation tables are current or voltage dependent and have a proportionality factor relative to the reference
        file

        :param parameter_list: A list of all Parameters() objects from the fit
        :param key: The Key to generate the saturation table for
        :param dc_bias_values: A list. The current or voltage values.
        :return: String. An LTSpice compatible saturation table
        """


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

    def entry_number_callback(self, checkstring):
        """
        Method to check whether something entered in an entry box is a valid float. Employs regex.
        Callback function for validate commands

        :param checkstring: The string to check
        :return: True if entered string is a valid float, False otherwise
        """
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

    def entry_to_float (self, number_string):
        """
        Auxilliary function to convert values from entry boxes to floats

        :param number_string: A string containing a float numeric value
        :return: A float from the string
        """
        try:
            return float(number_string)
        except:
            return None

    def start_GUI(self):
        """
        Method to start the GUI

        :return: None
        """
        self.root.mainloop()
