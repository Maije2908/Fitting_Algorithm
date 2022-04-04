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
from fitter import *

# print different System settings:
# operating system
if platform.system() == 'Linux':
    print('Linux detected.')
elif platform.system() == 'Windows':
    print('Windows detected.')
elif platform.system() == 'Darwin':
    print('Mac detected.')
else:
    print('No valid operating system detected!')

# start GUI
if __name__ == '__main__':
    #initialize iohandler prior to GUI, since GUI needs an instance of it, same goes for the fitter
    iohandler = IOhandler()
    fitter = Fitter()
    gui = GUI(iohandler, fitter)

    gui.start() #i was thinking to put the start method in the constructor, but that