# import packages
import tkinter as tk
from tkinter import filedialog
import config
import os
import re

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
        self.entry_saturation = None
        self.entry_value = None
        self.entry_resistance = None
        self.entry_prominence = None
        self.selected_s2p_files = None
        self.iohandler = iohandler_instance

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

    def create_drop_down(self):
        drop_down_var = tk.StringVar(self.root)

        option_menu = tk.OptionMenu(self.root, drop_down_var, *config.DROP_DOWN_ELEMENTS)
        max_drop_length = len(max(config.DROP_DOWN_ELEMENTS, key=len))
        option_menu.config(font=config.DROP_DOWN_FONT, width=max_drop_length + 5, height=config.DROP_DOWN_HEIGHT)
        option_menu.grid(column=0, row=0, columnspan=2, sticky=tk.N, )

    def create_specification_field(self):

        # validate command for inputs "register" is necessary so that the actual input is checked (would otherwise
        # update after input)
        vcmd = (self.root.register(self.entry_number_callback), "%P")

        # Headline
        label_spec = tk.Label(self.root, text="Specification", bg=config.BCKGND_COLOR)
        label_spec.config(font=config.HEADLINE_FONT)
        label_spec.grid(column=0, row=1, columnspan=2, sticky=tk.N, **config.SPEC_PADDING)

        # initial value
        label_value = tk.Label(self.root, text="F/C", bg=config.BCKGND_COLOR)
        label_value.config(font=config.ENTRY_FONT)
        label_value.grid(column=0, row=2, sticky=tk.W, **config.ENTRY_PADDING)

        self.entry_value = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_value.config(font=config.ENTRY_FONT)
        self.entry_value.grid(column=1, row=2, sticky=tk.W)

        # initial resistance value
        label_resistance = tk.Label(self.root, text="\u03A9", bg=config.BCKGND_COLOR)
        label_resistance.config(font=config.ENTRY_FONT)
        label_resistance.grid(column=0, row=3, sticky=tk.W, **config.ENTRY_PADDING)

        self.entry_resistance = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_resistance.config(font=config.ENTRY_FONT)
        self.entry_resistance.grid(column=1, row=3, sticky=tk.W)

        # Saturation Table
        label_saturation = tk.Label(self.root, text="Saturation Table", bg=config.BCKGND_COLOR)
        label_saturation.config(font=config.ENTRY_FONT)
        label_saturation.grid(column=0, row=4, sticky=tk.W, **config.ENTRY_PADDING)
        # endregion

        self.entry_saturation = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_saturation.config(font=config.ENTRY_FONT)
        self.entry_saturation.grid(column=1, row=4, sticky=tk.W, **config.ENTRY_PADDING)

        # Prominence
        label_prominence = tk.Label(self.root, text="Prominence in °", bg=config.BCKGND_COLOR)
        label_prominence.config(font=config.ENTRY_FONT)
        label_prominence.grid(column=0, row=5, sticky=tk.W, **config.ENTRY_PADDING)

        self.entry_prominence = tk.Entry(self.root, validate='all', validatecommand=(vcmd))
        self.entry_prominence.config(font=config.ENTRY_FONT)
        self.entry_prominence.grid(column=1, row=5, sticky=tk.W, **config.ENTRY_PADDING)

    def create_browse_button(self):
        browse_button = tk.Button(self.root, command=self.callback_browse_s2p_file, text="Select s2p File(s)")
        browse_button.config(font=config.ENTRY_FONT)
        browse_button.grid(column=0, row=6, sticky=tk.W, **config.ENTRY_PADDING)

    def create_run_button(self):
        browse_button = tk.Button(self.root, command=self.callback_run, text="Run (dummy)")
        browse_button.config(font=config.ENTRY_FONT)
        browse_button.grid(column=0, row=7, sticky=tk.W, **config.ENTRY_PADDING)

    ####################################################################################################################
    # Button commands

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



        return 0

    def callback_run(self):
        test = self.entry_value.get()
        return 0

    ####################################################################################################################
    # auxilliary functions

    def entry_number_callback(self, checkstring):
        # check function, invoked by the validationcommand of the entry-fields
        # returns TRUE if the value entried is a number; can have a leading + or - and can have ONE decimal point (.)
        # returns FALSE otherwise

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


    def start(self):
        self.root.mainloop()
