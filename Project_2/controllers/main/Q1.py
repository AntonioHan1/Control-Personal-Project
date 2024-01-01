import numpy as np

def find_P_Q(xdot):
      lr = 1.39
      lf = 1.55
      Ca = 20000
      Iz = 25854
      m = 1888.6
      g = 9.81
      A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,-2*Ca*(lf-lr)/(m*xdot)],[0,0,0,1],[0,-2*Ca*(lf-lr)/(Iz*xdot),2*Ca*(lf-lr)/Iz,-2*Ca*(lf**2+lr**2)/(Iz*xdot)]])
      B = np.array([[0,0],[2*Ca/m,0],[0,0],[2*Ca*(lf/Iz),0]])
      print(A)
      print(B)
      C = np.identity(4)
      P = np.hstack((B,A @ B,A @ A @ B,A @ A @ A @ B))
      Q = np.hstack((C,C @ A,C @ A @ A,C @ A @ A @ A))
      # print("xdot = ", str(xdot),":")
      # print("P:")
      # print(P)
      # print("Q:")
      # print(Q)
      
find_P_Q(2)
