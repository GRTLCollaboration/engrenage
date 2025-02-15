import numpy as np
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from core.grid import *
from bssn.tensoralgebra import *

class NoMatter :
    """Represents the matter that sources the Einstein equation - in this case none!"""

    def __init__(self) :
        
        # Details for the matter state variables
        self.NUM_MATTER_VARS = 0
        self.VARIABLE_NAMES = []
        self.PARITY = np.array([])
        self.ASYMP_POWER = np.array([])
        self.ASYMP_OFFSET = np.array([])
        self.indices = np.array([])
        self.matter_vars_set = True
    
    def get_emtensor(self, r, bssn_vars, background) :
    
        assert self.matter_vars_set, 'Matter vars not set'

        N = np.size(r) 
        empty_emtensor = EMTensor(N)             
            
        return empty_emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background) :

        assert self.matter_vars_set, 'Matter vars not set'
        
        return None
    
    # Set the matter vars and their derivs from the full state vector
    def set_matter_vars(self, state_vector, bssn_vars : BSSNVars, grid : Grid) :
        
        self.matter_vars_set = True
        