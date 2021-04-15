ARCH           ?= 50
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_$(ARCH) -std=c++11 -O3

all: presynaptic_scan

presynaptic_scan: kernel.o
	$(CXX) -o presynaptic_scan kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt

kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f presynaptic_scan kernel.o
