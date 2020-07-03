from devito.operator.profiling import PerfEntry
from examples.seismic.elastic_VTI.elastic_VTI_example import elastic_VTI_setup

from thematrix.common import check_norms, run_prepare, run_benchmark

class ForwardElasticVTI(object):

    # Problem setup
    params = ([(100, 100, 100)], [12], [{'rec1': 3.314084, 'rec2': 0.049992}])
    param_names = ['shape', 'space_order', 'norms']
    tn = 100

    # ASV parameters
    repeat = 1
    timeout = 900.0
    processes = 1
    
    
    
    
    def track_runtime(self, shape, space_order, norms):
        return self.summary.time
    track_runtime.unit = "runtime"

    def track_gflopss(self, shape, space_order, norms):
        return self.summary.gflopss
    track_gflopss.unit = "gflopss"

    def track_gpointss(self, shape, space_order, norms):
        return self.summary.gpointss
    track_gpointss.unit = "gpointss"