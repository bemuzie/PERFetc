from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "smth",
    version = "0.1",
    description = "testing smth in cython",
    author = "Denis Nesterov",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("smth",
        ["somecode.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-O3'])
    ]
)