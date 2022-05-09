# import packages
import tkinter as tk
from tkinter import filedialog
#maybe import later for logging to gui
#import tkinter.scrolledtext as scroll_text
#import logging
from fitter import *
import config
import os
import re
from tkinter import scrolledtext
from texthandler import *

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
        label_prominence = tk.Label(self.root, text="Prominence in °", bg=config.BCKGND_COLOR)
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

    def create_file_list(self):
        self.file_listbox = tk.Listbox(self.root)
        self.file_listbox.config(font=config.ENTRY_FONT)
        self.file_listbox.grid(column=3, row=0, rowspan=4, sticky=tk.W, **config.ENTRY_PADDING)

    def create_log_window(self):
        self.st = scrolledtext.ScrolledText(self.root, state='disabled')#, width=config.LOG_WIDTH,  height=config.LOG_HEIGHT)
        self.st.configure(font='TkFixedFont')
        self.st.grid(column=0,row=9,columnspan=3,sticky=tk.W,**config.ENTRY_PADDING)
        # self.st.pack()
        self.texthndl = Text_Handler(self.st)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.texthndl)
        self.logger.setLevel(logging.INFO)



    def insert_file_to_list(self, file):
        checkboxvalue = tk.BooleanVar()
        checkbox_to_insert = tk.Checkbutton(self.root,text = "Helloworld", variable=checkboxvalue)

        # self.file_listbox.insert(tk.END,checkbox_to_insert)




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

        #insert the file to the listbox here(TODO: not functional yet)
        self.insert_file_to_list(2)

        return 0


    #method to run the fitting algorithm, invoked when "run" button is pressed
    def callback_run(self):

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
            else:
                # self.logger.error('Something is wrong with the dropdown menu, please restart the application')
                raise Exception('Something is wrong with the dropdown menu, please restart the application')

        except Exception as e:
            #write exception to log
            self.logger.error(str(e) + '\n')
            #TODO: maybe do some robust error handling?
            raise





        try:
            # create a fitter instance (the logger instance needs to be passed to the constructor)
            self.fitter = Fitter(self.logger)

            # parse files to fitter
            self.fitter.set_file(self.iohandler.files[0])

            #calculate z21
            match shunt_series:
                case config.SHUNT_THROUGH:
                    self.fitter.calc_shunt_thru(config.Z0)
                case config.SERIES_THROUGH:
                    self.fitter.calc_series_thru(config.Z0)
                case _:
                    #This should NEVER be invoked (if so, there is something SERIOUSLY wrong with the radiobuttons)
                    raise Exception("Could not determine Shunt/Series Through, please re-select and try again")

            self.fitter.smooth_data()

            # parse specs to fitter
            self.fitter.set_specification(passive_nom, res, prom, sat, fit_type)

            self.fitter.get_main_resonance()
            self.fitter.get_resonances()

            self.fitter.create_nominal_parameters()
            #self.fitter.create_elements()
            self.fitter.start_fit()

            path_out = self.selected_s2p_files[0]
            self.iohandler.generate_Netlist_2_port(self.fitter, fit_type, path_out, '')


        except Exception as e:
            self.logger.error("ERROR: An Exception occurred during execution:")
            self.logger.error(str(e) + '\n')


        finally:
            self.fitter = None

        return 0

    ####################################################################################################################
    # auxilliary functions


    # check function, invoked by the validationcommand of the entry-fields
    # returns TRUE if the value entried is a number; can have a leading + or - and can have ONE decimal point (.)
    # returns FALSE otherwise
    def entry_number_callback(self, checkstring):
        # regular expression copied from: https://stackoverflow.com/questions/46116037/tkinter-restrict-entry-data-to-float
        regex = re.compile(r"(\+|\-)?[0-9.]*$")
        result = regex.match(checkstring)
        checkval = (checkstring == ""
                    or (checkstring.count('+') <= 1
                        and checkstring.count('-') <= 1
                        and checkstring.count('.') <= 1
                        and result is not None
                        and result.group(0) != ""))
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
