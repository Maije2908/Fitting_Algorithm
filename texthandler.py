

import logging
import tkinter
import threading
"""
"""


class Text_Handler(logging.Handler):
    """
        Texthandler for GUI output of log messages; copypasted

        This class allows you to log to a Tkinter Text or ScrolledText widget
        #see https://gist.github.com/moshekaplan/c425f861de7bbf28ef06 for reference
    """

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        """
        Method to emit text to the GUI scrolledtext widget
        Overrides method in logging.Handler
        :param record: Message to output
        :return: None
        """
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tkinter.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tkinter.END)
        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)