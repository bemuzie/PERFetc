from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "bilateral",
    version = "0.1",
    description = "nd bilateral filter",
    author = "Denis Nesterov",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ndbilateral",
        ["bilateral3d.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-O3'])
    ]
)