#Derivatives.py

import numpy as np
from source.uservariables import *

# For the derivation of the stencil coefficients see the notebook 
# DerivativeCalculations.ipynb in tests
            
# Derivative matrix class
# First make a class for the Heat Equation

class Derivatives :
    
    """
    Represents the derivatives on a grid defined by the points in r_vector, 
    with num_points_r points in total (including ghosts) over a range of values range_in_r. 
    Uses one sided derivatives at the end of the interval. These may or may not be overwritten in
    the boundary code. This code assumes 3 ghost cells by default.
    
    attributes: r_vector, range_in_r, num_points_r, log_factor (determines the ratio 
    by which the dx increases at each interval)
                
    methods: calculate the derivative matrices for the specified grid
    
    """
    
    # constructor function
    def __init__(self, a_r_vector, a_dr_vector):

        self.r_vector = a_r_vector
        self.dr_vector = a_dr_vector
        self.num_points_r = np.size(a_r_vector)
        
        # Find the log_factor from the dr values
        self.log_factor = (a_dr_vector[1] / a_dr_vector[0])
        # just check this factor is consistent at the end points too, then assume it is constant
        check = (a_dr_vector[self.num_points_r-1]/a_dr_vector[self.num_points_r-2])
        #print(check, self.log_factor)
        assert (self.log_factor == check), 'log factor not constant in r vector'
        
        # calculate some useful values for the stencils
        self.calculate_useful_values()
        
        # use them for the derivatives
        self.d1_matrix = self.get_first_derivative_matrix()
        self.d2_matrix = self.get_second_derivative_matrix()
        self.d_advec_matrix_left = self.get_left_advection_derivative_matrix()  
        self.d_advec_matrix_right = self.get_right_advection_derivative_matrix()  
        self.d_dissipation_matrix = self.get_dissipation_derivative_matrix()

    def get_first_derivative_matrix(self) :
        
        d1_matrix = np.zeros([self.num_points_r,self.num_points_r])

        indices = np.arange(self.num_points_r)
        for idx_i in indices :
            
            # Factors of the grid spacing
            h = self.dr_vector[idx_i]
            one_over_h = 1.0 / h
            
            # Populate non zero matrix elements
            for idx_j in indices :
                
                if (idx_i == (idx_j+2)) :
                    d1_matrix[idx_i,idx_j] = self.d1_stencil[0] * one_over_h
                
                elif (idx_i == (idx_j+1)) :
                    d1_matrix[idx_i,idx_j] = self.d1_stencil[1] * one_over_h

                elif(idx_i == idx_j) :
                    d1_matrix[idx_i,idx_j] = self.d1_stencil[2] * one_over_h                    
                    
                elif (idx_i == (idx_j-1)) :
                    d1_matrix[idx_i,idx_j] = self.d1_stencil[3] * one_over_h

                elif (idx_i == (idx_j-2)) :
                    d1_matrix[idx_i,idx_j] = self.d1_stencil[4] * one_over_h
                
                else :
                    d1_matrix[idx_i,idx_j] = 0.0
                
        # Fix the 2 points at each end with one sided derivatives
        for idx in np.array([0,1]) :
            # left hand stencil
            if idx == 1 :
                d1_matrix[idx,idx-1] = 0.0
            d1_matrix[idx,idx] = self.advec_d1_stencil_right[0] * one_over_h
            d1_matrix[idx,idx+1] = self.advec_d1_stencil_right[1] * one_over_h
            d1_matrix[idx,idx+2] = self.advec_d1_stencil_right[2] * one_over_h
            d1_matrix[idx,idx+3] = self.advec_d1_stencil_right[3] * one_over_h        
            
        for idx in np.array([self.num_points_r - 2, self.num_points_r - 1]) :
            # right hand stencil
            d1_matrix[idx,idx-3] = self.advec_d1_stencil_left[0] * one_over_h
            d1_matrix[idx,idx-2] = self.advec_d1_stencil_left[1] * one_over_h
            d1_matrix[idx,idx-1] = self.advec_d1_stencil_left[2] * one_over_h        
            d1_matrix[idx,idx  ] = self.advec_d1_stencil_left[3] * one_over_h             
            if idx == self.num_points_r - 2 :
                d1_matrix[idx,idx+1] = 0.0    
        return d1_matrix         
        
    def get_second_derivative_matrix(self) :
        
        d2_matrix = np.zeros([self.num_points_r,self.num_points_r])

        indices = np.arange(self.num_points_r)
        for idx_i in indices :
            
            # Factors of the grid spacing
            h = self.dr_vector[idx_i]
            one_over_h2 = 1.0 / (h*h)
            
            # Populate non zero matrix elements
            for idx_j in indices :

                if (idx_i == (idx_j+2)) :
                    d2_matrix[idx_i,idx_j] = self.d2_stencil[0] * one_over_h2
                
                elif (idx_i == (idx_j+1)) :
                    d2_matrix[idx_i,idx_j] = self.d2_stencil[1] * one_over_h2

                elif(idx_i == idx_j) :
                    d2_matrix[idx_i,idx_j] = self.d2_stencil[2] * one_over_h2                    
                    
                elif (idx_i == (idx_j-1)) :
                    d2_matrix[idx_i,idx_j] = self.d2_stencil[3] * one_over_h2

                elif (idx_i == (idx_j-2)) :
                    d2_matrix[idx_i,idx_j] = self.d2_stencil[4] * one_over_h2
                
                else :
                    d2_matrix[idx_i,idx_j] = 0.0
                
        # Fix the 2 points at each end with one sided derivatives
        for idx in np.array([0,1]) :
            # left hand stencil
            if idx == 1 :
                d2_matrix[idx,idx-1] = 0.0
            d2_matrix[idx,  idx] = self.advec_d2_stencil_right[0] * one_over_h2
            d2_matrix[idx,1+idx] = self.advec_d2_stencil_right[1] * one_over_h2
            d2_matrix[idx,2+idx] = self.advec_d2_stencil_right[2] * one_over_h2
            d2_matrix[idx,3+idx] = self.advec_d2_stencil_right[3] * one_over_h2         
            
        for idx in np.array([self.num_points_r - 2, self.num_points_r - 1]) :
            # right hand stencil
            d2_matrix[idx,idx-3] = self.advec_d2_stencil_left[0] * one_over_h2
            d2_matrix[idx,idx-2] = self.advec_d2_stencil_left[1] * one_over_h2
            d2_matrix[idx,idx-1] = self.advec_d2_stencil_left[2] * one_over_h2        
            d2_matrix[idx,idx  ] = self.advec_d2_stencil_left[3] * one_over_h2             
            if idx == self.num_points_r - 2 :
                d2_matrix[idx,idx+1] = 0.0    
        return d2_matrix 

    def get_left_advection_derivative_matrix(self) :
        
        advec_matrix = np.zeros([self.num_points_r,self.num_points_r])

        indices = np.arange(self.num_points_r)
        for idx_i in indices :
            
            # Factors of the grid spacing
            h = self.dr_vector[idx_i]
            one_over_h = 1.0 / h
            
            # Populate non zero matrix elements
            for idx_j in indices :
                
                if (idx_i == (idx_j+3)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_left[0] * one_over_h
                
                elif (idx_i == (idx_j+2)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_left[1] * one_over_h

                elif(idx_i == idx_j+1) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_left[2] * one_over_h
                    
                elif (idx_i == (idx_j)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_left[3] * one_over_h
                
                else :
                    advec_matrix[idx_i,idx_j] = 0.0
                
        # Zero the advective terms for the ghost cells where no left cells exist
        for idx in np.array([0,1,2]) :
            # left hand stencil
            advec_matrix[idx, :] = 0.0           
    
        return advec_matrix      

    def get_right_advection_derivative_matrix(self) :
        
        advec_matrix = np.zeros([self.num_points_r,self.num_points_r])

        indices = np.arange(self.num_points_r)
        for idx_i in indices :
            
            # Factors of the grid spacing
            h = self.dr_vector[idx_i]
            one_over_h = 1.0 / h
            
            # Populate non zero matrix elements
            for idx_j in indices :
                
                if (idx_i == (idx_j)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_right[0] * one_over_h
                
                elif (idx_i == (idx_j-1)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_right[1] * one_over_h

                elif(idx_i == idx_j-2) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_right[2] * one_over_h                    
                    
                elif (idx_i == (idx_j-3)) :
                    advec_matrix[idx_i,idx_j] = self.advec_d1_stencil_right[3] * one_over_h
                
                else :
                    advec_matrix[idx_i,idx_j] = 0.0
                
        # Zero the advective terms for the ghost cells where no right cells exist
        for idx in np.array([self.num_points_r - 3, self.num_points_r - 2, self.num_points_r - 1]) :
            # left hand stencil
            advec_matrix[idx, :] = 0.0           
    
        return advec_matrix      

    def get_dissipation_derivative_matrix(self) :
        
        diss_matrix = np.zeros([self.num_points_r,self.num_points_r])

        indices = np.arange(self.num_points_r)
        for idx_i in indices :
            
            # Factors of the grid spacing for the derivative and 2^N factor
            h = self.dr_vector[idx_i]
            one_over_h6 = 1.0 / (h**6.0) / 64.0
            
            # Populate non zero matrix elements
            for idx_j in indices :

                if   (idx_i == (idx_j+3)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[0] * one_over_h6
                
                elif (idx_i == (idx_j+2)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[1] * one_over_h6

                elif (idx_i == idx_j+1) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[2] * one_over_h6 
                    
                elif (idx_i == (idx_j)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[3] * one_over_h6

                elif (idx_i == (idx_j-1)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[4] * one_over_h6
                    
                elif (idx_i == (idx_j-2)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[3] * one_over_h6

                elif (idx_i == (idx_j-3)) :
                    diss_matrix[idx_i,idx_j] = self.dissipation_derivative_stencil[4] * one_over_h6
                    
                else :
                    diss_matrix[idx_i,idx_j] = 0.0
               
        
        # zero the dissipation in the ghost cells
        for idx in np.array([0,1,2,
                             self.num_points_r - 3,self.num_points_r - 2, self.num_points_r - 1]) :
            diss_matrix[idx,:] = 0.0
                 
        return diss_matrix     
    
    
    def calculate_useful_values(self) :
        
        c = self.log_factor
        c2 = c*c
        c3 = c2 * c
        c4 = c2 * c2
        c5 = c2 * c3
        c6 = c3 * c3
        c7 = c3 * c4
        c8 = c4 * c4
        c9 = c5 * c4
        oneplusc = 1.0 + c
        oneplusc2 = 1.0 + c2
        onepluscplusc2 = 1.0 + c + c*c
        
        # Finite difference coefficients
        
        # Centered first derivative (fourth order)
        Ap2 = - 1.0 / ( c2 * oneplusc * oneplusc2 * onepluscplusc2 )
        Ap1 = oneplusc / (c2 * onepluscplusc2 )
        A0 = 2.0 * (c - 1.0) / c
        Am1 = - c4 * Ap1
        Am2 = - c8 * Ap2
        
        self.d1_stencil = np.array([Am2, Am1, A0, Ap1, Ap2])
        
        # Centered second derivative (fourth order)
        Bp2 = 2.0 * (1.0 - 2.0*c2 ) / ( c3 * oneplusc * oneplusc * oneplusc2 * onepluscplusc2 )
        Bp1 = 2.0 * (2.0*c - 1.0) * oneplusc / ( c3 * onepluscplusc2 )
        B0  = 2.0 * (1.0 - c - 5.0*c2 - c3 + c4) / ( c2 * oneplusc * oneplusc )
        Bm1 = 2.0 * (2.0 - c) * c * oneplusc / onepluscplusc2
        Bm2 = 2.0 * c7 * (c2 - 2.0) / ( c2 * oneplusc * oneplusc * oneplusc2 * onepluscplusc2 )

        self.d2_stencil = np.array([Bm2, Bm1, B0, Bp1, Bp2])      
        
        # one sided (right) first derivative (third order)
        Cp3 = 1 / c / onepluscplusc2
        Cp2 = - onepluscplusc2 / c / oneplusc
        Cp1 = onepluscplusc2
        C0 = - c2 * (c3 + 3 * c2 + 4 * c + 3) / (c3 + 2 * c2 + 2 * c + 1)

        self.advec_d1_stencil_right = np.array([C0, Cp1, Cp2, Cp3])      
        
        # one sided (left) first derivative (third order)
        D0 = (3*c3 + 4*c2 + 3*c + 1) / (c3 + 2*c2 + 2*c + 1 )
        Dm1 = - onepluscplusc2 
        Dm2 = c2 * onepluscplusc2 / (oneplusc)
        Dm3 = -c5 / onepluscplusc2

        self.advec_d1_stencil_left = np.array([Dm3, Dm2, Dm1, D0])       

        # one sided (right) second derivative (third order)
        Ep3 = -2 * (c+2) / c2 / onepluscplusc2 / oneplusc
        Ep2 =  2 * (onepluscplusc2 + 1) / c2 / oneplusc
        Ep1 = - 2 * (c2 + 2 * c + 2) / c / oneplusc
        E0 = 2 * c * (c2 + 2 * c + 3) / onepluscplusc2 / oneplusc

        self.advec_d2_stencil_right = np.array([E0, Ep1, Ep2, Ep3])      
        
        # one sided (left) second derivative (third order)
        F0 = 2 * (3*c2 + 2*c + 1 ) / c2 / oneplusc / onepluscplusc2
        Fm1 = -2 * (2*c2 + 2*c + 1) / c2 / oneplusc
        Fm2 = 2 * (c2 + onepluscplusc2) / oneplusc / c2
        Fm3 = - 2 * c2 * (2*c + 1) / oneplusc / onepluscplusc2

        self.advec_d2_stencil_left = np.array([Fm3, Fm2, Fm1, F0])        
        
        # Kreiss Oliger dissipation coefficients (6th derivative)
        Gp3 = 1.0
        Gp2 = 1.0
        Gp1 = 1.0
        G0  = 1.0
        Gm1 = 1.0
        Gm2 = 1.0
        Gm3 = 1.0
        
        self.dissipation_derivative_stencil = np.array([Gm3, Gm2, Gm1, G0, Gp1, Gp2, Gp3])
        
        # Weights for interpolation at the end points, using the innermost valid
        # 4 grid points, label the values at these f1, f2, f3, f4
        # WA is for the point at r = -dx/2 - dx/c (ignore parity)
        WA1 = (c8 + c7 - 2*c5 - 2*c4 + 2*c2 + c - 1)/ (c6 * onepluscplusc2)
        WA2 = 1
        WA3 = 1
        WA4 = 1

        # WB is the point at r = -dx/2 - dx/c - dx/c/c (ignore parity)
        WB1 = (c9 - c7 - 2*c6 - c5 + c4 +3*c3 + c2 - c - 1) / c9
        WB2 = 1
        WB3 = 1
        WB4 = 1
        
    #End of Derivatives class