
# The IOhandler class shall be in charge of file handling. It will be instanced in the main function and shall be given
# to the GUI as an instance. It shall hold pointers to instances of the S2pFile class and shall be able to manage them
# in order to get the option to delete loaded files from the list and to load new files to the list

from sNpfile import *
import skrf as rf

class IOhandler:

    def __init__(self):
        self.files = []
        #self.number_of_files = 0 <- obsolete! info is in the list


    def load_file(self, path, ind_cap_cmc):

        for actual_path in path:
            ntwk = rf.Network(actual_path)
            touch = rf.Touchstone(actual_path) #I don't know if this is necessary, in payer's program it was used, but the
            #network class of skrf stores all the information necessary, so I would consider it a waste of memory

            # generate class for storing the data in and write to array
            newfile = SNpFile(ntwk, ntwk.name)
            self.files.append(newfile)

        return 0