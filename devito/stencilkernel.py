from itertools import chain
from devito.iteration import Iteration
from devito.expression import Expression
from devito.tools import filter_ordered
import cgen as c

__all__ = ['StencilKernel']


class StencilKernel(object):
    """Code generation class, alternative to Propagator"""

    def __init__(self, stencils, name='Kernel'):
        self.name = name

        # Ensure we always deal with Expression lists
        stencils = stencils if isinstance(stencils, list) else [stencils]
        self.expressions = [Expression(s) for s in stencils]

        # Wrap expressions with Iterations according to dimensions
        for i, expr in enumerate(self.expressions):
            newexpr = expr
            for d in reversed(expr.dimensions):
                newexpr = Iteration(newexpr, d, d.size)
            self.expressions[i] = newexpr

        # TODO: Merge Iterations iff outermost variables agree

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stenicl kernel to a set of data objects"""
        raise NotImplementedError("StencilKernel - Codegen and apply() missing")

    @property
    def signature(self):
        """List of data objects that define the kernel signature

        :returns: List of unique data objects required by the kernel
        """
        signatures = [e.signature for e in self.expressions]
        return filter_ordered(chain(*signatures))

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
        header_vars = [c.Pointer(c.POD(v.dtype, '%s_vec' % v.name))
                       for v in self.signature]
        header = c.Extern("C", c.FunctionDeclaration(
            c.Value('int', self.name), header_vars))
        cast_shapes = [(v, ''.join(['[%d]' % d for d in v.shape[1:]]))
                       for v in self.signature]
        casts = [c.Initializer(c.POD(v.dtype, '(*%s)%s' % (v.name, shape)),
                               '(%s (*)%s) %s' % (c.dtype_to_ctype(v.dtype),
                                                  shape, '%s_vec' % v.name))
                 for v, shape in cast_shapes]
        body = [e.ccode for e in self.expressions]
        ret = [c.Statement("return 0")]
        return c.FunctionBody(header, c.Block(casts + body + ret))
