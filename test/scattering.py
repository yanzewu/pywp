import numpy as np
import sys
import unittest

sys.path.append('..')
from pywp.scattering.impl import scatter1d
from pywp import mgrid
from pywp.util import build_pe_tensor
from pywp.potential.test import Tully1_1D


# flat test
np.set_printoptions(linewidth=1000, precision=3)

class TestScattering(unittest.TestCase):

    def assertDictAlmostEqual(self, a, b, decimal=2):
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            np.testing.assert_almost_equal(a[k], b[k], decimal=decimal, verbose=False)

    def test_flat(self):
        print('Testing flat')
        self.assertDictAlmostEqual(scatter1d(np.zeros((10,1,1)), mgrid[-1:1:10j], 1000, 0.001, 0, side='both', incoming_side='left')[0], {'left':[0], 'right':[1]})
        self.assertDictAlmostEqual(scatter1d(np.zeros((10,1,1)), mgrid[-1:1:10j], 1000, 0.001, 0, side='both', incoming_side='right')[0], {'left':[1], 'right':[0]})
        self.assertDictAlmostEqual(scatter1d(np.zeros((10,1,1)), mgrid[-1:1:10j], 1000, 0.001, 0, side='left', incoming_side='left')[0], {'left':[1]})
        self.assertDictAlmostEqual(scatter1d(np.zeros((10,1,1)), mgrid[-1:1:10j], 1000, 0.001, 0, side='right', incoming_side='right')[0], {'right':[1]})

    def test_1state_potential(self):
        print('Testing potential')
        grid = mgrid[-5:5:252j]
        H = np.tanh(grid.build()[0]) * 0.02
        self.assertDictAlmostEqual(scatter1d(H[:,None,None], grid, 1000, 0.01, 0, side='both', incoming_side='left')[0], {'left':[1], 'right':[0]})
        self.assertDictAlmostEqual(scatter1d(H[:,None,None], grid, 1000, 0.01, 0, side='left', incoming_side='left')[0], {'left':[1]})
        self.assertDictAlmostEqual(scatter1d(H[:,None,None], grid, 1000, 0.01, 0, side='both', incoming_side='right')[0], {'left':[1], 'right':[0]})

    def test_multistate_potential(self):
        print('Testing multistate')
        grid = mgrid[-5:5:252j]
        H = np.zeros((grid.size, 3,3))
        H[:,0,0] = np.tanh(grid.build()[0]) * 0.05
        H[:,1,1] = -H[:,0,0] - 0.09
        self.assertDictAlmostEqual(scatter1d(H, grid, 1000, 0.01, 0, side='both', incoming_side='left')[0], {'left':[1,0,0], 'right':[0,0,0]})
        self.assertDictAlmostEqual(scatter1d(H, grid, 1000, 0.01, 0, side='both', incoming_side='right')[0], {'left':[1,0,0], 'right':[0,0,0]})
        self.assertDictAlmostEqual(scatter1d(H, grid, 1000, 0.01, 1, side='both', incoming_side='left')[0], {'left':[0,0,0], 'right':[0,1,0]})
        self.assertDictAlmostEqual(scatter1d(H, grid, 1000, 0.01, 1, side='both', incoming_side='right')[0], {'left':[0,0,0], 'right':[0,1,0]})

    def test_tully1(self):
        print('Testing tully1')
        grid = mgrid[-5:5:252j]
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), grid, 2000, 0.025, 0, side='both', incoming_side='left')[0], {'left':[0,0], 'right':[0.16,0.84]}, 2)
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), mgrid[-10:10:1020j], 2000, 0.1, 0, side='both', incoming_side='left')[0], {'left':[0,0], 'right':[0.49,0.51]}, 2)
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), mgrid[-10:10:1020j], 2000, 0.225, 1, side='both', incoming_side='right')[0], {'left':[0.28,0.72], 'right':[0,0]}, 2)
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), grid, 2000, 0.025, 1, side='both', incoming_side='left')[0], {'left':[0,0], 'right':[0.73,0.27]}, 2)
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), mgrid[-10:10:1020j], 2000, 0.1, 1, side='both', incoming_side='left', adiabatic_boundary=True)[0], {'left':[0,0], 'right':[0.55,0.45]}, 2)
        self.assertDictAlmostEqual(scatter1d(Tully1_1D(), mgrid[-10:10:1020j], 2000, 0.225, 0, side='both', incoming_side='right')[0], {'left':[0.74,0.26], 'right':[0,0]}, 2)

    def test_coherent(self):
        print('Testing coherent')
        H = lambda R: build_pe_tensor(np.ones_like(R[0])*1e-5, 0.01, symmetric=True)
        self.assertDictAlmostEqual(scatter1d(H, mgrid[-5:5:124j], 1000, 0.02, [0.5**0.5, 0.5**0.5], adiabatic_boundary=True)[0], {'left':[0,0], 'right':[0,1]})

    def test_tully3(self):
        print('Testing tully3')
        self.assertDictAlmostEqual(scatter1d(lambda R: build_pe_tensor(
                                    6e-4*np.ones_like(R[0]), 
                                    0.1*np.exp(0.9*R[0]) * (R[0] < 0) + 0.1 * (2 - np.exp(-0.9*R[0])) * (R[0] >= 0),
                                    symmetric=True), 
                                mgrid[-20:10:1020j], 2000, 0.15625, 0, adiabatic_boundary=True)[0], {'left':[0.24,0.34], 'right':[0.43,0]})

    def test_tully1_adiabatic(self):
        print('Testing tully1 (adiabatic)')

        grid = mgrid[-5:5:1020j]
        R = grid.build()
        H = Tully1_1D().get_H(R)
        E, U = np.linalg.eigh(H)
        A = 0.01
        B = 1.6
        C = 0.005
        dHdR = build_pe_tensor(A*B*np.exp(-np.abs(B*R[0])),-2*R[0]*C*np.exp(-R[0]**2), symmetric=True)
        dHdR_ad = np.transpose(U, (0,2,1)) @ dHdR @ U
        D = build_pe_tensor(np.zeros_like(R[0]), np.abs(dHdR_ad[:,0,1]) / (E[:,1]-E[:,0]), symmetric=True, antiherm=True)
        Emat = build_pe_tensor(E[:,0], 0, E[:,1])
        self.assertDictAlmostEqual(scatter1d(Emat, grid, 2000, 0.1, 0, side='both', incoming_side='left', drvcoupling=D)[0], {'left':[0,0], 'right':[0.51,0.49]})


if __name__ == '__main__':
    unittest.main()
    