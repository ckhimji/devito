from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag
from examples.seismic import PointSource, Receiver


def src_rec(v, tau, model, geometry):
    """
    Source injection and receiver interpolation
    """
    ts = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)

    # The source injection term
    src_xx = src.inject(field=tau[0, 0].forward, expr=src * ts)
    src_zz = src.inject(field=tau[-1, -1].forward, expr=src * ts)
    src_term = src_xx + src_zz
    if model.grid.dim == 3:
        src_yy = src.inject(field=tau[1, 1].forward, expr=src * ts)
        src_term += src_yy

    # Create interpolation expression for receivers -- needs to be fixed
    rec_term1 = rec1.interpolate(expr=tau[-1, -1])
    rec_term2 = rec2.interpolate(expr=div(v))

    return src_term + rec_term1 + rec_term2


def ForwardOperator(model, geometry, space_order=4, time_order = 1, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """

    v = VectorTimeFunction(name='v', grid=model.grid,
                           space_order=space_order, time_order=time_order)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             space_order=space_order, time_order=time_order)

    ## Symbolic physical parameters used from the model
    irho = model.irho

    c11 = model.c11
    c33 = model.c33
    c44 = model.c44
    c66 = model.c66
    c13 = model.c13

    # Model critical timestep computed from CFL condition for VTI media
    ts = model.grid.stepping_dim.spacing 
    damp = model.damp.data

    # Particle Velocity for each direction
    u_vx = Eq(v[0].forward, damp*v[0] + damp*ts*irho*(tau[0,0].dx + tau[1,0].dy + tau[2,0].dz) )
    u_vy = Eq(v[1].forward, damp*v[1] + damp*ts*irho*(tau[0,1].dx + tau[1,1].dy + tau[1,2].dz) )
    u_vz = Eq(v[2].forward, damp*v[2] + damp*ts*irho*(tau[0,2].dx + tau[2,1].dy + tau[2,2].dz) )

    # Stress for each direction in VTI Media:
    u_txx = Eq(tau[0,0].forward, damp*tau[0,0] + damp*ts*(c11*v[0].forward.dx + c11*v[1].forward.dy - 2*c66*v[1].forward.dy + c13*v[2].forward.dz) )
    u_tyy = Eq(tau[1,1].forward, damp*tau[1,1] + damp*ts*(c11*v[0].forward.dx - 2*c66*v[0].forward.dx + c11*v[1].forward.dy + c13*v[2].forward.dz) )
    u_tzz = Eq(tau[2,2].forward, damp*tau[2,2] + damp*ts*(c13*v[0].forward.dx + c13*v[1].forward.dy + c33*v[2].forward.dz) )

    u_txz = Eq(tau[0,2].forward, damp*tau[0,2] + damp*ts*(c44*v[2].forward.dx + c44*v[0].forward.dz) )
    u_tyz = Eq(tau[1,2].forward, damp*tau[1,2] + damp*ts*(c44*v[2].forward.dy + c44*v[1].forward.dz) )
    u_txy = Eq(tau[0,1].forward, damp*tau[0,1] + damp*ts*(c66*v[1].forward.dx + c66*v[0].forward.dy) )
    
    stencil = [u_vx, u_vy, u_vz, u_txx, u_tyy, u_tzz, u_txz, u_tyz, u_txy]
    srcrec = src_rec(v, tau, model, geometry)
    pde = stencil + srcrec
    op = Operator(pde, subs=model.spacing_map, name="ForwardElasticVTI")

    # Substitute spacing terms to reduce flops
    return op
