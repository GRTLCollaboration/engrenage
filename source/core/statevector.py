# statevector.py

import numpy as np
from bssn.bssnstatevariables import *

class StateVector :
    """Creates the full state vector from the bssn vars and the given matter class."""

    def __init__(self, a_matter) :
        
        # Details for the matter state variables
        self.NUM_VARS = NUM_BSSN_VARS + a_matter.NUM_MATTER_VARS
        self.VARIABLE_NAMES = np.concatenate([BSSN_VARIABLE_NAMES, a_matter.VARIABLE_NAMES])
        self.PARITY = np.concatenate([BSSN_PARITY, a_matter.PARITY])
        self.ASYMP_POWER = np.concatenate([BSSN_ASYMP_POWER, a_matter.ASYMP_POWER])
        self.ASYMP_OFFSET = np.concatenate([BSSN_ASYMP_OFFSET, a_matter.ASYMP_OFFSET])
        self.ALL_INDICES = np.arange(self.NUM_VARS, dtype=np.uint8)
                               
    def print_variable_details(self) :
        
        print(self.VARIABLE_NAMES)