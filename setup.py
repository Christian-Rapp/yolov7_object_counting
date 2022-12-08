import setuptools
import numpy
from Cython.Build import cythonize

setuptools.setup(
    name             = 'seq-nms-counting',
    version          = '0.0.1',
    description      = 'Seq NMS for Object Counting.',
    url              = 'https://github.com/Christian-Rapp/yolov7_object_counting',
    author           = 'Christian Rapp',
    packages         = setuptools.find_packages(),
    ext_modules      = cythonize("compute_overlap.pyx"),
    include_dirs     = [numpy.get_include()],
    setup_requires   = ["cython>=0.28", "numpy>=1.14.0"]
)