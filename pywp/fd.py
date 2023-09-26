
import numpy as np
from scipy.special import factorial

def drv_kernel(order:int=1, accuracy:int=1) -> np.ndarray:
    """ Return a 1D array of derivative kernel. See https://en.wikipedia.org/wiki/Finite_difference_coefficient for the formula.
    order: Order of the derivative: Only 1/2;
    accuracy: The accuracy of the derivative. Positive integers.

    Returns an numpy array with length = accuracy*2 + 1.
    """
    n = accuracy
    s = np.zeros(2*n+1)
    p = np.arange(1, n+1)
    
    if order == 1:
        s[n+1:] = (-1)**(p+1)*factorial(n)**2/(p*factorial(n-p)*factorial(n+p))
        s[:n] = -s[-1:n:-1]
    
    elif order == 2:
        s[n+1:] = 2*(-1)**(p+1)*factorial(n)**2/(p**2*factorial(n-p)*factorial(n+p))
        s[:n] = s[-1:n:-1]
        s[n] = -2*np.sum(1/np.arange(1,n+1)**2)

    else:
        raise ValueError('order can only be 1 or 2')
    
    return s


def drv_matrix(length:int, order:int=1, accuracy:int=1):
    """ Return a 2D square matrix for derivative, known as the difference operator in quantum mechanics.

    length: Size of the matrix will be length x length;
    order: Order of the derivative: Only 1/2;
    accuracy: The accuracy of the derivative. Positive integers.
    """

    m = np.zeros(length*length)
    s = drv_kernel(order, accuracy)
    
    # diagonal
    m[::length+1] = s[accuracy]

    # off diagonal
    for j in range(1,accuracy+1):
        m[j:(length-j)*length:length+1] = s[accuracy+j]
        m[length*j::length+1] = s[accuracy-j]

    return m.reshape((length, length))


def laplacian_kernel(accuracy:int=1, dim:int=2, uniform=None):
    """ Return a N-D matrix as the convolution kernel for the laplacian operator (\\nabla^2).
    
    By default, will simply paste 1D derivative operator along the axis. Special options exist for dim == 2 or 3.
    - dim == 2: 
        - uniform == 1: uses the 9-point stencil with \\gamma=1/3 (see https://en.wikipedia.org/wiki/Nine-point_stencil);
        - uniform == 2: uses the 9-point stencil with \\gamma=1/2; 
    - dim == 3:
        - uniform == 1: uses the 19-point stencil (R. C. O'Reilly and J. M. Beck, Int. J. Numer. Meth. Engng 2006; 00:1-16);
        - uniform == 2: uses the 27-point stencil (the same paper);

    """
    if not uniform:
        s = drv_kernel(2, accuracy)
        x = np.zeros((len(s),)*dim)
        for j in range(dim):
            indexer = (accuracy,) * j + (slice(None),) + (accuracy,) * (dim-j-1)
            x[indexer] += s
        return x

    if uniform:
        assert uniform in (1,2), "Invalid uniform option"
        assert dim in (2,3), "Only support 2D and 3D"
        if dim == 2:
            gamma = 1/3 if uniform == 1 else 1/2
            return (1-gamma)*np.array([[0,1,0],[1,-4,1],[0,1,0]]) + \
                gamma * np.array([[0.5,0,0.5],[0,-2,0],[0.5,0,0.5]])
        
        elif dim == 3:
            if uniform == 1:
                return np.array([0,1,0,1,2,1,0,1,0,
                                 1,2,1,2,-24,2,1,2,1,
                                 0,1,0,1,2,1,0,1,0]).reshape((3,3,3))/6
            elif uniform == 2:
                return np.array([2,3,2,3,6,3,2,3,2,
                                 3,6,3,6,-88,6,3,6,3,
                                 2,3,2,3,6,3,2,3,2]).reshape((3,3,3))/26