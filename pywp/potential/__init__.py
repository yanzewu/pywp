
class Potential:
    """ Interface of potential class.
    """

    def __init__(self, dim=2, kdim=2, *params):
        """ `potential_params` in command args will be passed to `params`.
        """
        self.dim = dim
        self.kdim = kdim

    def get_H(self, R):
        """ 
        Get the Hamiltonian in position basis.

        R will be (N, N, ...) array-like instance, representing the meshgrid position grid.
        Should return an array-like object, with size (N, N, ..., nel, nel)
        """
        return None

    def get_kdim(self):
        """ Return kinetic dimension
        """
        return self.kdim

    def get_dim(self):
        """ Return the electronic dimension
        """
        return self.dim

    def has_get_phase(self):
        return False

    def get_phase(self, R):
        return 1


def get_potential(name):

    from os.path import dirname, basename, isfile, join
    import glob
    import importlib

    fname, pname = name.split('.', 1)

    modules = [basename(f)[:-3] for f in glob.glob(join(dirname(__file__), "*.py")) if isfile(f)]
    if fname not in modules:
        return None
    else:
        try:
            p = getattr(importlib.import_module('.' + fname, package=__name__), pname)
        except AttributeError:
            return None
            
        if issubclass(p, Potential):
            return p
        else:
            return None


