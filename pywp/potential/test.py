
import numpy as np

from . import Potential

class Nothing(Potential):

    def __init__(self):
        super().__init__()

    def get_H(self, R):
        H = np.zeros((R[0].shape[0], R[0].shape[1], 2, 2))
        H[:,:,0,0] = -0.1
        H[:,:,1,1] = 0.1
        return H

class Tully1_1D(Potential):
    def __init__(self, *params):

        self.A = 0.01
        self.B = 1.6
        self.C = 0.005
        super().__init__()

    def get_H(self, R):
        
        H = np.zeros((R[0].shape[0], 2, 2))
        H[:,0,0] = self.A*(1 - np.exp(-self.B*R[0])) * (R[0] >= 0) - self.A*(1 - np.exp(self.B*R[0])) * (R[0] < 0)
        H[:,1,1] = -H[:,0,0]
        H[:,0,1] = self.C*np.exp(-R[0]**2)
        H[:,1,0] = H[:,0,1]

        return H
    
    def get_kdim(self):
        return 1


class Tully1(Potential):

    def __init__(self, *params):

        self.A = 0.01
        self.B = 1.6
        self.C = 0.005
        super().__init__()

    def get_H(self, R):
        
        H = np.zeros((R[0].shape[0], R[0].shape[1], 2, 2))
        H[:,:,0,0] = self.A*(1 - np.exp(-self.B*R[0])) * (R[0] >= 0) - self.A*(1 - np.exp(self.B*R[0])) * (R[0] < 0)
        H[:,:,1,1] = -H[:,:,0,0]
        H[:,:,0,1] = self.C*np.exp(-R[0]**2)
        H[:,:,1,0] = H[:,:,0,1]

        return H

    def has_get_phase(self):
        return True


class Flat(Potential):

    def __init__(self, A, B, W):
        self.A = A
        self.B = B
        self.W = W
        super().__init__()

    def get_H(self, R):

        from scipy.special import erf

        H = np.zeros((R[0].shape[0], R[0].shape[1], 2, 2), dtype=complex)
        Theta = np.pi/2*(erf(self.B*R[0]) + 1)
        H[:,:,0,0] = -self.A * np.cos(Theta)
        H[:,:,1,1] = -H[:,:,0,0]
        H[:,:,0,1] = self.A * np.sin(Theta) * np.exp(1j*self.W*R[1])
        H[:,:,1,0] = np.conj(H[:,:,0,1])
        return H

    def has_get_phase(self):
        return True

    def get_phase(self, R):
        return np.exp(1j*self.W*R[1])
        