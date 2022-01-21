
import numpy as np
import sys

sys.path.append('..')
import pywp

class Tully1(pywp.Potential):

    def __init__(self, *params):

        self.A = 0.01
        self.B = 1.0
        self.C = 0.005

    def get_H(self, R):
        
        H = np.zeros((R[0].shape[0], R[0].shape[1], 2, 2))
        H[:,:,0,0] = self.A*(1 - np.exp(-self.B*R[0])) * (R[0] >= 0) - self.A*(1 - np.exp(self.B*R[0])) * (R[0] < 0)
        H[:,:,1,1] = -H[:,:,0,0]
        H[:,:,0,1] = self.C*np.exp(-R[0]**2)
        H[:,:,1,0] = H[:,:,0,1]

        return H

myapp = pywp.Application(Tully1)
myapp.parse_args()
myapp.run()
