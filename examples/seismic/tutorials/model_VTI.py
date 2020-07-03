import sympy as sp
from sympy import sin, Abs
import numpy as np

from examples.seismic.model import initialize_damp, PhysicalDomain, GenericModel
from devito.builtins import initialize_function, gaussian_smooth, mmax

__all__ = ['ModelElasticVTI']

class ModelElasticVTI(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats, optional
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : float or array
        P-wave velocity in km/s.
    vs : float or array
        S-wave velocity in km/s.
    rho : float or array, optional
        Density in kg/cm^3 (rho=1 for water).
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    gamma : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.   
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.
    nbl : int, optional
        The number of absorbing layers for boundary damping.
    dtype : np.float32 or np.float64
        Defaults to 32.

    The `ModelElasticVTI` provides a symbolic data objects for the
    creation of seismic wave propagation operators in VTI media:

    damp : Function, optional
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, vs, rho, epsilon, delta, gamma, theta=None, phi=None,
                 nbl=20, fs = False, dtype=np.float32, subdomains=(), bcs="mask", grid=None, **kwargs):
        super(ModelElasticVTI, self).__init__(origin, spacing, shape, space_order, nbl,
                                                dtype, subdomains, grid=grid, bcs=bcs, fs=fs)
        
        self.maxvp = np.max(vp)
        # Create square slowness of the wave as symbol `m`
        self._vp = self._gen_phys_param(vp, 'vp', space_order)
        self._vs = self._gen_phys_param(vs, 'vs', space_order)
        
        self.rho = self._gen_phys_param(rho, 'rho', space_order, is_param=True)
        self.irho = self._gen_phys_param(1./rho, 'irho', space_order, is_param=True)

        # Additional parameter fields for VTI operators
        self.epsilon = self._gen_phys_param(epsilon, 'epsilon', space_order)
        self.scale = 1 if epsilon is None else np.sqrt(1 + 2 * np.max(epsilon))
        self.delta = self._gen_phys_param(delta, 'delta', space_order)
        self.gamma = self._gen_phys_param(gamma, 'gamma', space_order)
        self.theta = self._gen_phys_param(theta, 'theta', space_order)
        if self.grid.dim > 2:
            self.phi = self._gen_phys_param(phi, 'phi', space_order)

        #Elastic Coefficients for a VTI medium

        # Thomsen's stiffness coefficients in elastic VTI media from The stiffness matrix, otherwise known as the elastic modulus matrix
        # these coefficients are computed by ModelElasticVTI
        #c11 = ro*(1+2*eps)*vp**2
        #c33 = ro*vp**2
        #c44 = ro*vs**2
        #c66 = ro*(1+2*gamma)*vs**2
        #f = 1 - vs**2/vp**2
        #c13 = ro*vp**2*sp.sqrt(f*(f+2*delta))-ro*vs**2

        self.c11 = self._gen_phys_param(rho*(1+2*epsilon)*vp**2, 'c11', space_order, is_param=True)
        self.c33 = self._gen_phys_param(rho*vp**2, 'c33', space_order, is_param=True)
        self.c44 = self._gen_phys_param(rho*vs**2, 'c44', space_order, is_param=True)
        self.c66 = self._gen_phys_param(rho*(1+2*gamma)*vs**2, 'c66', space_order, is_param=True)
        self.c13 = self._gen_phys_param(rho*vp**2*sp.sqrt((1 - vs**2/vp**2)*((1 - vs**2/vp**2)+2*delta))-rho*vs**2, 'c13', space_order, is_param=True)

    @property
    def _max_vp(self):
        return mmax(self.vp)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*self._max_vp))
        return self.dtype("%.3e" % dt)

    @property
    def vp(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            if vp.shape == self.vp.shape:
                self.vp.data[:] = vp[:]
            elif vp.shape == self.shape:
                initialize_function(self._vp, vp, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model of size" % vp.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.vp.shape))
        else:
            self._vp.data = vp

    @property
    def vs(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vs

    @vs.setter
    def vs(self, vs):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(vs, np.ndarray):
            if vs.shape == self.vs.shape:
                self.vs.data[:] = vs[:]
            elif vs.shape == self.shape:
                initialize_function(self._vs, vs, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model of size" % vs.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.vs.shape))
        else:
            self._vs.data = vs

    @property
    def m(self):
        return 1 / (self.vp * self.vp)
    
    @property
    def smooth(self, physical_parameters, sigma=5.0):
        """
        Apply devito.gaussian_smooth to model physical parameters.

        Parameters
        ----------
        physical_parameters : string or tuple of string
            Names of the fields to be smoothed.
        sigma : float
            Standard deviation of the smoothing operator.
        """
        model_parameters = self.physical_params()
        for i in physical_parameters:
            gaussian_smooth(model_parameters[i], sigma=sigma)
        return
