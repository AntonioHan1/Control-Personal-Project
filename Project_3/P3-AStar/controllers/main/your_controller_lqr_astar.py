# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,-2*Ca*(lf-lr)/(m*xdot)],[0,0,0,1],[0,-2*Ca*(lf-lr)/(Iz*xdot),2*Ca*(lf-lr)/Iz,-2*Ca*(lf**2+lr**2)/(Iz*xdot)]])
        B = np.array([[0,0],[2*Ca/m,0],[0,0],[2*Ca*(lf/Iz),0]])
        C = np.identity(4)
        D = np.zeros((4,1))
        A_dis,B_dis,C_dis,D_dis,dt = signal.cont2discrete((A,B,C,D), delT, method='zoh')
        Q = 0.2*np.identity(4)
        R = 1*np.identity(2)
        S = np.matrix(linalg.solve_discrete_are(A_dis, B_dis, Q, R))
        K = -np.matrix(linalg.inv(B_dis.T@S@B_dis+R)@(B_dis.T@S@A_dis))
        K = K[0]
        
        _, node = closestNode(X, Y, trajectory)
        forwardIndex = 85
        
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1],trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 = (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)



        except:
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1],trajectory[-1,0]-trajectory[node,0])
            e1 = (Y - trajectory[-1,1])*np.cos(psiDesired) - (X - trajectory[-1,0])*np.sin(psiDesired)

            
        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot
        
        state = np.array([[e1],[e1dot],[e2],[e2dot]])
        
        delta = (K @ state)
        delta = delta[0,0]
        
        # ---------------|Longitudinal Controller|-------------------------

        desired_V = 10
        e_V =  desired_V - xdot
        e_V_P = e_V
        # update the sum of previous error and last error
        self.sum_pre_e_V += e_V
        e_V_I = self.sum_pre_e_V
        e_V_D = e_V-self.last_e_V
        self.last_e_V = e_V
        
        #parameters
        K_p_F = 180
        K_i_F = 8
        K_d_F = 28
        
        
        F = K_p_F*e_V_P + K_i_F*e_V_I + K_d_F*e_V_D

        # Return all states and calculated control inputs (F, delta) and obstacle position
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
