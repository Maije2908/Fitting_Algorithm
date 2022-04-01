

# this file is thought to hold the data of one s2p file and should not provide much more functionality
# it may be important to NOT declare the data this class holds as global, since we need to take care about namespaces
# this class should be invoked by the IOhandler class and all data should be handled by the IOhandler

# in a way the S2pFile class should only provide the "memory" for the IOhandler and should make scaleability and file
# handling easier


class SNpFile:

    def __init__(self, sNpdata, sNname):
        self.data = sNpdata
        self.name = sNname

    def set_data(self, sNpdata, sNname):
        self.data = sNpdata
        self.name = sNname
        return 0

    def get_data(self):
        return self.data