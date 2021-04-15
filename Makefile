ARCH           ?= 50
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_$(ARCH) -std=c++11 -O3 -rdc=true

all: presynaptic_scan

presynaptic_scan: kernel_dlink.o
	$(CXX) -o presynaptic_scan kernel.o kernel_dlink.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

kernel_dlink.o: kernel.o
	$(NVCC) -arch sm_$(ARCH) -dlink -o kernel_dlink.o kernel.o -lcudart -lcudadevrt

kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f presynaptic_scan kernel.o kernel_dlink.o
