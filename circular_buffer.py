import numpy as np

class CircularBuffer:
    """
    Class implement circular buffer.
    """
    def __init__(self, length : int, data_type : np.dtype) -> None:
        """
        Constructor CircularBuffer

        Parameters
        ----------
        lenght : unsigned int
            Buffor length (size).
        
        data_type : numpy.dtype
            Type of data stored in buffer.
        """
        self.length     = length
        self.data_type  = data_type
        self.buffer     = np.zeros((length,1), dtype=data_type)
        self.pointer    = 0


    def push(self, element : np.dtype,) -> None:
        """
        Push element on top FIFO list
        
        Parameters
        ----------
        element : self.data_type
            Element to add
        """
        self.pointer = self.pointer % self.length
        self.buffer[self.pointer ] = element
        self.pointer += 1 


    def get_buffor(self) -> np.array:
        """
        Fuction return buffer in FIFO sequence 
        
        Returns
        -------
        buffer : numpy.array
            Buffer in FIFO sequence        
        """
        return np.roll(self.buffer, self.length-self.pointer)[::-1]