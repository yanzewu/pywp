
from .pywp import CheckPoint, abs2, PhysicalParameter


class Operator:
    """ Base class for operators.
    """

    def __call__(self, para:PhysicalParameter, checkpoint:CheckPoint):
        ...


class PositionFunc(Operator):
    """ Evaluates an operator which is a function of position on each electronic states.
    f_n = \int_{|Psi_n(R)|^2 f(R)dR} / [norm]
    """

    def __init__(self, *op, normalization='state'):
        """ op: An (N1 x ... Nk) array, or a callable taking position and returning such array.
            normalization: 'state'/'full'/'none'. \
                - state: Normalization is performed on each state. f_n = <psi_n|A|psi_n> / <psi_n|psi_n>
                - full: Normalization is performed on the full wavefunction. f_n = <psi_n|A|psi_n> / <psi|psi>
                - none: No normalization is performed. f_n = <psi_n|A|psi_n>. Used when there is absorbing factors.
        """
        self.op = op
        self.normalization = normalization

    def __call__(self, para: PhysicalParameter, checkpoint: CheckPoint):
        
        abspsi2 = abs2(checkpoint.psiR)
        nk = len(para.R)
        result = []

        if self.normalization == 'state':
            population = checkpoint.backend.sum(abspsi2, tuple(range(nk)))
        elif self.normalization == 'full':
            population = checkpoint.backend.sum(abspsi2)
        else:
            population = 1/para.dA
        
        for op in self.op:
            if callable(op):
                _op = op(para.R)
            else:
                _op = op

            result.append(checkpoint.backend.sum(abspsi2*_op[...,None], tuple(range(nk))) / population)

        if len(self.op) == 1:
            return result[0]
        else:
            return result


class PositionElectronicFunc(Operator):
    """ Evaluates an operator which is a function if both position and each electronic states.
        f = \int_{Psi_n(R)\Psi^*_m(R) f_{mn}(R)dR} / \int_{|Psi(R)|^2dR}
        Returns an scalar for each partition.
    """
    def __init__(self, *op, normalization='full'):
        """ op: An (N1 x ... Nk x Nel x Nel) array, or a callable taking position and returning such array.
            normalization: 'state'/'full'/'none'. \
                - full: Normalization is performed on the full wavefunction. f_n = <psi_n|A|psi_n> / <psi|psi>
                - none: No normalization is performed. f_n = <psi_n|A|psi_n>. Used when there is absorbing factors.
        """
        self.op = op
        self.normalization = normalization

    def __call__(self, para: PhysicalParameter, checkpoint: CheckPoint):
        
        result = []
        if self.normalization == 'full':
            abspsi2 = abs2(checkpoint.psiR)
            population = checkpoint.backend.sum(abspsi2)
        else:
            population = 1/para.dA
        
        for op in self.op:
            if callable(op):
                _op = op(para.R)
            else:
                _op = op

            result.append(checkpoint.backend.vdot(checkpoint.psiR, _op @ checkpoint.psiR[...,None]) / population)

        if len(self.op) == 1:
            return result[0]
        else:
            return result
        
        
class Position(Operator):
    """ Evaluates the position (R) as a vector.
    Returns: <R>, <R_n> (Nel x Nk vector).
    <R> = \int_{|Psi(R)|^2 RdR} / \int_{|Psi(R)|^2dR}
    <R_n> = \int_{|Psi_n(R)|^2 RdR} / \int_{|Psi_n(R)|^2dR}
    """

    def __call__(self, para: PhysicalParameter, checkpoint: CheckPoint):
        
        _backend = checkpoint.backend
        nk = len(para.R)

        abspsi2 = abs2(checkpoint.psiR)
        population = _backend.sum(abspsi2, tuple(range(nk)))

        aveR = _backend.array([_backend.sum(abspsi2*R_[...,None], axis=tuple(range(nk))) / population for R_ in para.R])
        
        return aveR @ population / _backend.sum(population), aveR


class Momentum(Operator):
    """ Evaluates the momentum (P) as a vector.
    Returns: <P>, <P_n> (Nel x Nk vector).
    <P> = \int_{|Psi(P)|^2 PdP} / \int_{|Psi(P)|^2dP}
    <P_n> = \int_{|Psi_n(P)|^2 PdP} / \int_{|Psi_n(P)|^2dP}
    """
    
    def __call__(self, para:PhysicalParameter, checkpoint:CheckPoint):

        _backend = checkpoint.backend
        nk = len(para.R)

        abspsip2 = abs2(checkpoint.psiK)
        population = _backend.sum(abspsip2, axis=tuple(range(nk)))

        aveK = _backend.array([_backend.sum(abspsip2*K_[...,None], axis=tuple(range(nk))) / population for K_ in para.K])

        return aveK @ population / _backend.sum(population), aveK


class KineticEnergy(Operator):
    """ Evaluates the kinetic energy (P^2/2M).
    Returns: <KE>, <KE_n> (Nel vector).
    <KE> = \int_{|Psi(P)|^2 (P^2/2M) dP} / \int_{|Psi(P)|^2dP}
    <KE_n> = \int_{|Psi_n(P)|^2 (P^2/2M) dP} / \int_{|Psi_n(P)|^2dP}
    """
    
    def __call__(self, para:PhysicalParameter, checkpoint:CheckPoint):

        _backend = checkpoint.backend
        nk = len(para.R)

        abspsip2 = abs2(checkpoint.psiK)
        population = _backend.sum(abspsip2, axis=tuple(range(nk)))

        aveKE = _backend.sum(abspsip2*para.KE[...,None], axis=tuple(range(nk))) / population

        return _backend.dot(aveKE, population) / _backend.sum(population), aveKE
    