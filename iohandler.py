
# The IOhandler class shall be in charge of file handling. It will be instanced in the main function and shall be given
# to the GUI as an instance. It shall hold pointers to instances of the S2pFile class and shall be able to manage them
# in order to get the option to delete loaded files from the list and to load new files to the list

from s2pfile import *

class IOhandler:

    def __init__(self):
        self.files = []
        #self.number_of_files = 0
        self.testvalue = 1


    def load_file(self, path, ind_cap_cmc):
        newfile = S2pFile()
        self.files.append(newfile)
        return 0