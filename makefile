#### NOT CURRENTLY WORKING. MISSING PYTORCH HEADER FILES. POSSIBLY C NOT SUPPORTED?

# compiles and installs extensions
# installs main package

CUDA_ARCH=-gencode arch=compute_30,code=sm_30\
		   -gencode arch=compute_35,code=sm_35\
		   -gencode arch=compute_50,code=sm_50\
		   -gencode arch=compute_52,code=sm_52\
		   -gencode arch=compute_60,code=sm_60\
		   -gencode arch=compute_61,code=sm_61\
	       -gencode arch=compute_70,code=sm_70
NMS=lib/nms/src/cuda
ROIALIGN=lib/roialign/roi_align/src/cuda
HEADERS=/home/ubuntu/pytorch/torch/lib/include

build: $(ROIALIGN)/crop_and_resize_kernel.cu.o # $(NMS)/nms_kernel.cu.o
	python lib/nms/build.py
	python lib/nms/setup.py install
	python lib/roialign/build.py
	python lib/roialign/setup.py install
	python setup.py install

$(ROIALIGN)/crop_and_resize_kernel.cu.o:
	echo 'Compiling crop_and_resize kernels by nvcc...'
	cd $(ROIALIGN);	nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu\
	    -x cu -Xcompiler "-I$(HEADERS)" -fPIC $(CUDA_ARCH)

$(NMS)/nms_kernel.cu.o:
	echo "Compiling nms kernels by nvcc..."
	cd $(NMS); nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC $(CUDA_ARCH)
