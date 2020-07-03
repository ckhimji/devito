import numpy as np
from argparse import ArgumentParser

from devito import configuration
from devito.logger import info
from elastic_VTI_solvers.wavesolver import ElasticVTIWaveSolver
from examples.seismic import setup_geometry
from model_VTI import ModelElasticVTI




def elastic_VTI_setup(origin=(0., 0., 0.), spacing=(15.0, 15.0, 15.0), shape=(50, 50, 50), space_order=4, vp= 2.0 , vs=1.0, rho=1.8, 
        epsilon=0.25, delta = 0.10, gamma = 0.05,
        nbl=10, tn=500., constant=False, **kwargs):

    model = ModelElasticVTI(origin=origin, spacing=spacing, shape=shape, space_order=space_order,
    vp=vp, vs=vs, rho=rho, epsilon=epsilon, delta = delta, gamma = gamma, nbl = nbl,    
                    dtype = kwargs.pop('dtype', np.float32), **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ElasticVTIWaveSolver(model, geometry, space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False, **kwargs):

    solver = elastic_VTI_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                           space_order=space_order, constant=constant, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, v, tau, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings,
            [rec1, rec2, v, tau])


def test_elastic_VTI():
    _, _, _, [rec1, rec2, v, tau] = run()
    norm = lambda x: np.linalg.norm(x.data.reshape(-1))
    assert np.isclose(norm(rec1), 23.7273, atol=1e-3, rtol=0)
    assert np.isclose(norm(rec2), 0.99306, atol=1e-3, rtol=0)


if __name__ == "__main__":
    description = ("Example script for a set of elastic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("-opt", default="advanced",
                        choices=configuration._accepted['opt'],
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    args = parser.parse_args()

    # 2D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    # 3D preset parameters
    else:
        shape = (150, 150, 150)
        spacing = (10.0, 10.0, 10.0)
        tn = 1250.0

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant)
