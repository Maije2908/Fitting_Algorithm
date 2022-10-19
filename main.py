#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================================================
# Created By  :
# Created Date:
# Course:
#
# =====================================================================================================================

import os
#    main packages and author information
import sys

#   information of author printed if console is used
__author__ = ' '
__copyright__ = 'Copyright 2021, ' + str(os.path.basename(sys.argv[0]))
__credits__ = [' ']
__version__ = '1.0.0'
__email1__ = ' '
__email2__ = ' '
__status__ = 'Draft'
__course__ = ''
__date__ = ''
__description__ = 'Parameter-Fit for inductors using lmfit'

#   import packages
import platform


# import my packages
from GUI import *
from iohandler import *

# start GUI
if __name__ == '__main__':
    #initialize iohandler prior to GUI, since GUI needs an instance of it
    iohandler = IOhandler()

    gui = GUI(iohandler)

    gui.start_GUI()