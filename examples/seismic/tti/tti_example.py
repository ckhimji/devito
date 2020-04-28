from argparse import ArgumentParser
from devito import configuration
from examples.seismic import demo_model, setup_geometry
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              space_order=4, nbl=10, preset='layers-tti', **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing,
                       space_order=space_order, nbl=nbl, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbl=10,
        kernel='centered', **kwargs):

    solver = tti_setup(shape, spacing, tn, space_order, nbl, **kwargs)

    rec, u, v, summary = solver.forward(autotune=autotune, kernel=kernel)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-d", "--shape", default=(50, 50, 50), type=int, nargs="+",
                        help="Determine the grid size")
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'staggered'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-opt", default="advanced",
                        choices=configuration._accepted['opt'],
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    args = parser.parse_args()

    preset = 'layers-tti-noazimuth' if args.azi else 'layers-tti'

    # preset parameters
    shape = args.shape
    ndim = len(shape)
    spacing = tuple(ndim * [10.0])
    tn = 750. if ndim < 3 else 250.

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune,
        opt=args.opt, kernel=args.kernel, preset=preset)
