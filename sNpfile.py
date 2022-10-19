



class SNpFile:
    """
    This class shall hold the data of an sNpfile (Touchstone file) and does not have much more functionality
    Instances of this class are managed by the IOhandler
    """

    def __init__(self, sNpdata, sNname):
        self.data = sNpdata
        self.name = sNname

    def set_data(self, sNpdata, sNname):
        """
        setter method for the SNpFile class, sets the data

        :param sNpdata: A skrf.Network() class
        :param sNname: The name of the file
        :return: None
        """
        self.data = sNpdata
        self.name = sNname

    def get_data(self):
        """
        Getter method for the SNpFile class
        :return: Data of the File. A skrf.Network() class.
        """
        return self.data