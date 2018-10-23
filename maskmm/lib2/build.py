from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 \
	   -gencode arch=compute_70,code=sm_70 "

sources = ['src/nms.c']
headers = ['src/nms.h']
defines = []
with_cuda = False
extra_objects = []

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.c']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    extra_objects = ['src/cuda/nms_kernel.cu.o']
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()


setup(
        name='nms',
        ext_modules=[
            CUDAExtension(
                    name='nms',
                    sources=['nms.cpp', 'nms.cu'],
                    extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['-O2']})
        ],
        cmdclass={
            'build_ext': BuildExtension
        })