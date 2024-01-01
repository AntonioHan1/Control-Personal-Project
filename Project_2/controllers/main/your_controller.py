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
        distance, idx = closestNode(X,Y, trajectory)
        if (len(trajectory) - idx-30) >= 1:
            refer_point_pos = trajectory[idx+30]
        else:
            refer_point_pos = trajectory[len(trajectory)-1]
        e_psi = wrapToPi(np.arctan2(refer_point_pos[1]-Y,refer_point_pos[0]-X)-psi)
        
        e1 = distance
        e1_dot = ydot+xdot*wrapToPi(psi - np.arctan2(refer_point_pos[1]-Y,refer_point_pos[0]-X))
        # This is a way to calculate e1_dot, but I found another way and try to use it below.
        
        # ref_point_close = trajectory[idx]
        # e1_dot = np.sqrt((ref_point_close[1]-Y-xdot*np.sin(psi)-ydot*np.cos(psi))**2+(ref_point_close[0]-X-xdot*np.cos(psi)+ydot*np.sin(psi))**2)-e1
        
        # It was just the difference between e1 at current and the next time step.
        # It also work, and I'm able to tune it to complete the cycle within 348s.
        # But it is not very fast, so I still use the previous method.

        e2 = e_psi
        e2_dot = psidot
        
        A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,-2*Ca*(lf-lr)/(m*xdot)],[0,0,0,1],[0,-2*Ca*(lf-lr)/(Iz*xdot),2*Ca*(lf-lr)/Iz,-2*Ca*(lf**2+lr**2)/(Iz*xdot)]])
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*(lf/Iz)]])
        Bunch_with_K = signal.place_poles(A, B, [-1,-2,-3,-4])
        K = Bunch_with_K.gain_matrix

        delta = (K @ np.array([[e1],[e1_dot],[e2],[e2_dot]]))[0][0]

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
        K_i_F = 0.001
        K_d_F = 0
        
        
        F = K_p_F*e_V_P + K_i_F*e_V_I + K_d_F*e_V_D
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
