# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        self.last_e_V =0
        self.sum_pre_e_V = 0
        self.last_e_psi =0
        self.sum_pre_e_psi = 0
        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        
        # ---------------|Lateral Controller|-------------------------
        _, idx = closestNode(X,Y, trajectory)
        if (len(trajectory) - idx-10) >= 1:
            refer_point_pos = trajectory[idx+10]
        else:
            refer_point_pos = trajectory[len(trajectory)-1]
        e_psi = wrapToPi(np.arctan2(refer_point_pos[1]-Y,refer_point_pos[0]-X)-psi)
        e_psi_P = e_psi
        # update the sum of previous error and last error
        self.sum_pre_e_psi += e_psi
        e_psi_I = self.sum_pre_e_psi
        e_psi_D = e_psi-self.last_e_psi
        self.last_e_psi = e_psi
        
        #parameters
        K_p_delta = 5
        K_i_delta = 0
        K_d_delta = 0
        
        delta = K_p_delta*e_psi_P + K_i_delta*e_psi_I + K_d_delta*e_psi_D

        # ---------------|Longitudinal Controller|-------------------------
        desired_V = 35
        e_V =  desired_V - xdot
        e_V_P = e_V
        # update the sum of previous error and last error
        self.sum_pre_e_V += e_V
        e_V_I = self.sum_pre_e_V
        e_V_D = e_V-self.last_e_V
        self.last_e_V = e_V
        
        #parameters
        K_p_F = 40
        K_i_F = 0
        K_d_F = 0
        
        
        F = K_p_F*e_V_P + K_i_F*e_V_I + K_d_F*e_V_D
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
