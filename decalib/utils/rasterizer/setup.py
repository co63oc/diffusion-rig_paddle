import paddle
import os
import setuptools
os.environ['CC'] = 'gcc-7'
os.environ['CXX'] = 'gcc-7'
USE_NINJA = os.getenv('USE_NINJA') == '1'
paddle.utils.cpp_extension.setup(name='standard_rasterize_cuda',
    ext_modules=[paddle.utils.cpp_extension.CUDAExtension(sources=[
    'standard_rasterize_cuda.cpp', 'standard_rasterize_cuda_kernel.cu'])])
