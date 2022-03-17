# import packages
import tkinter as tk

import config

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
    def __init__(self):
        # Window config
        self.root: tk.Tk = tk.Tk()
        self.root.wm_title('Fitting Program V2')
        self.root.config(bg='#FFFFFF')

        # here starts the creation of the widgets
        self.create_drop_down()
        self.create_specification_field()

    def create_drop_down(self):
        drop_down_var = tk.StringVar(self.root)

        option_menu = tk.OptionMenu(self.root, drop_down_var, *config.DROP_DOWN_ELEMENTS)
        max_drop_length = len(max(config.DROP_DOWN_ELEMENTS, key=len))
        option_menu.config(font=config.DROP_DOWN_FONT, width=max_drop_length+5, height=config.DROP_DOWN_HEIGHT)
        option_menu.grid(column=0, row=0, columnspan=2, sticky=tk.N,)

    def create_specification_field(self):
        '''
        # Headline
        label_spec = tk.Label(self.root, text="Specification", bg=config.BCKGND_COLOR)
        label_spec.config(width=config.SPEC_WIDTH, height=config.SPEC_HEIGHT, font=config.HEADLINE_FONT)
        label_spec.grid(column=0, row=1, columnspan=2, sticky=tk.N, **config.SPEC_PADDING)

        # initial value
        label_value = tk.Label(self.root, text="F/C", bg=config.BCKGND_COLOR)
        label_value.config(width=config.ENTRY_WIDTH, height=config.ENTRY_HEIGHT, font=config.ENTRY_FONT)
        label_value.grid(column=0, row=2, sticky=tk.W, **config.ENTRY_PADDING)

        entry_value = tk.Entry(self.root)
        entry_value.config(width=config.ENTRY_WIDTH, font=config.ENTRY_FONT)
        entry_value.grid(column=1, row=2, sticky=tk.W)

        # initial resistance value
        label_resistance = tk.Label(self.root, text="\u03A9", bg=config.BCKGND_COLOR)
        label_resistance.config(width=config.ENTRY_WIDTH, height=config.ENTRY_HEIGHT, font=config.ENTRY_FONT)
        label_resistance.grid(column=0, row=3, sticky=tk.W, **config.ENTRY_PADDING)

        entry_resistance = tk.Entry(self.root)
        entry_resistance.config(width=config.ENTRY_WIDTH, font=config.ENTRY_FONT)
        entry_resistance.grid(column=1, row=3, sticky=tk.W)

        # Saturation Table
        label_saturation = tk.Label(self.root, text="Saturation Table", bg=config.BCKGND_COLOR)
        label_saturation.config(width=config.ENTRY_WIDTH, height=config.ENTRY_HEIGHT, font=config.ENTRY_FONT)
        label_saturation.grid(column=0, row=4, sticky=tk.W, **config.ENTRY_PADDING)

        entry_saturation = tk.Entry(self.root)
        entry_saturation.config(width=config.ENTRY_WIDTH, font=config.ENTRY_FONT)
        entry_saturation.grid(column=1, row=4, sticky=tk.W, **config.ENTRY_PADDING)

        # Prominence
        label_prominence = tk.Label(self.root, text="Prominence in °", bg=config.BCKGND_COLOR)
        label_prominence.config(width=config.ENTRY_WIDTH, height=config.ENTRY_HEIGHT, font=config.ENTRY_FONT)
        label_prominence.grid(column=0, row=5, sticky=tk.W, **config.ENTRY_PADDING)

        entry_prominence = tk.Entry(self.root)
        entry_prominence.config(width=config.ENTRY_WIDTH, font=config.ENTRY_FONT)
        entry_prominence.grid(column=1, row=5, sticky=tk.W, **config.ENTRY_PADDING)
        '''

        # Headline
        label_spec = tk.Label(self.root, text="Specification", bg=config.BCKGND_COLOR)
        label_spec.config(font=config.HEADLINE_FONT)
        label_spec.grid(column=0, row=1, columnspan=2, sticky=tk.N, **config.SPEC_PADDING)

        # initial value
        label_value = tk.Label(self.root, text="F/C", bg=config.BCKGND_COLOR)
        label_value.config(font=config.ENTRY_FONT)
        label_value.grid(column=0, row=2, sticky=tk.W, **config.ENTRY_PADDING)

        entry_value = tk.Entry(self.root)
        entry_value.config(font=config.ENTRY_FONT)
        entry_value.grid(column=1, row=2, sticky=tk.W)

        # initial resistance value
        label_resistance = tk.Label(self.root, text="\u03A9", bg=config.BCKGND_COLOR)
        label_resistance.config(font=config.ENTRY_FONT)
        label_resistance.grid(column=0, row=3, sticky=tk.W, **config.ENTRY_PADDING)

        entry_resistance = tk.Entry(self.root)
        entry_resistance.config(font=config.ENTRY_FONT)
        entry_resistance.grid(column=1, row=3, sticky=tk.W)



        # Saturation Table
        label_saturation = tk.Label(self.root, text="Saturation Table", bg=config.BCKGND_COLOR)
        label_saturation.config(font=config.ENTRY_FONT)
        label_saturation.grid(column=0, row=4, sticky=tk.W, **config.ENTRY_PADDING)
        # endregion

        entry_saturation = tk.Entry(self.root)
        entry_saturation.config(font=config.ENTRY_FONT)
        entry_saturation.grid(column=1, row=4, sticky=tk.W, **config.ENTRY_PADDING)

        # Prominence
        label_prominence = tk.Label(self.root, text="Prominence in °", bg=config.BCKGND_COLOR)
        label_prominence.config(font=config.ENTRY_FONT)
        label_prominence.grid(column=0, row=5, sticky=tk.W, **config.ENTRY_PADDING)

        entry_prominence = tk.Entry(self.root)
        entry_prominence.config(font=config.ENTRY_FONT)
        entry_prominence.grid(column=1, row=5, sticky=tk.W, **config.ENTRY_PADDING)


    def start(self):
        self.root.mainloop()
