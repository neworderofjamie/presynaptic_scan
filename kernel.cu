// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define NUM_POPULATIONS 50
#define SEED 124

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }


template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

struct MergedPresynapticUpdateGroup
{
    float *inSyn;
    unsigned int *srcSpkCnt;
    unsigned int *srcSpk;
    float *weight;
};

// Host globals
unsigned int mergedGroupStartID[NUM_POPULATIONS];
MergedPresynapticUpdateGroup mergedGroups[NUM_POPULATIONS];

// Device globals
__device__ unsigned int d_mergedGroupStartID[NUM_POPULATIONS];
__device__ __constant__  MergedPresynapticUpdateGroup d_mergedGroups[NUM_POPULATIONS];

__global__ void presynapticUpdateIdleThreads()
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    unsigned int lo = 0;
    unsigned int hi = NUM_POPULATIONS;
    while(lo < hi) {
        const unsigned int mid = (lo + hi) / 2;
        if(id < d_mergedGroupStartID[mid]) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }
    struct MergedPresynapticUpdateGroup *group = &d_mergedGroups[lo - 1]; 
    const unsigned int groupStartID = d_mergedGroupStartID[lo - 1];
    const unsigned int lid = id - groupStartID;

    if(lid < group->srcSpkCnt[0]) {
        const unsigned int preInd = group->srcSpk[lid];
        atomicAdd(&group->inSyn[preInd], group->weight[preInd]);
    }
}


//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(size_t count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, size_t count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, size_t count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    const unsigned int popSize = 5000;
    const unsigned int numSpikes = 500;
    const unsigned int blockSize = 32;
    const bool oracle = true;

    const unsigned int paddedPopSize = ((popSize + blockSize - 1) / blockSize) * blockSize;
    std::cout << "Padded pop size:" << paddedPopSize << std::endl;

    CHECK_CUDA_ERRORS(cudaSetDevice(0));

    cudaEvent_t updateStart;
    cudaEvent_t updateEnd;
    CHECK_CUDA_ERRORS(cudaEventCreate(&updateStart));
    CHECK_CUDA_ERRORS(cudaEventCreate(&updateEnd));

    std::mt19937 rng;
    std::uniform_int_distribution<unsigned int> spikeDist(0, popSize - 1);
    std::normal_distribution<float> weightDist(0.0f, 0.25f);

    HostDeviceArray<float> inSyn[NUM_POPULATIONS];
    std::vector<float> correctInSyn[NUM_POPULATIONS];

    unsigned int startThread = 0;
    for(unsigned int i = 0; i < NUM_POPULATIONS; i++) {
        // Resize and zero correct insyn vector
        correctInSyn[i].resize(popSize, 0.0f);

        // Allocate memory
        inSyn[i] = allocateHostDevice<float>(popSize);
        auto srcSpkCnt = allocateHostDevice<unsigned int>(1);
        auto srcSpk = allocateHostDevice<unsigned int>(popSize);
        auto weight = allocateHostDevice<float>(popSize);

        // Zero inSyn
        std::fill_n(&inSyn[i].first[0], popSize, 0.0f);
        

        // Generate random spikes
        srcSpkCnt.first[0] = numSpikes;
        std::generate_n(&srcSpk.first[0], numSpikes, [&rng, &spikeDist]() { return spikeDist(rng); });

        // Generate weights
        std::generate_n(&weight.first[0], popSize, [&rng, &weightDist]() { return weightDist(rng); });

        // Calculate correct output
        for(unsigned int j = 0; j < numSpikes; j++) {
            const unsigned int ind = srcSpk.first[j];
            correctInSyn[i][ind] += weight.first[ind];
        }

        // Upload
        hostToDeviceCopy(inSyn[i], popSize);
        hostToDeviceCopy(srcSpkCnt, 1, true);
        hostToDeviceCopy(srcSpk, popSize, true);
        hostToDeviceCopy(weight, popSize, true);

        // Build struct with device pointers
        mergedGroups[i].inSyn = inSyn[i].second;
        mergedGroups[i].srcSpk = srcSpk.second;
        mergedGroups[i].srcSpkCnt = srcSpkCnt.second;
        mergedGroups[i].weight = weight.second;

        // Calculate static start ID
        mergedGroupStartID[i] = oracle ? startThread : (i * paddedPopSize);
        
        // Sum padded spikes
        startThread += ((numSpikes + blockSize - 1) / blockSize) * blockSize;        
    }

    // Copy merged group structures to symbols
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedGroups, &mergedGroups[0], sizeof(MergedPresynapticUpdateGroup) * NUM_POPULATIONS));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedGroupStartID, &mergedGroupStartID[0], sizeof(unsigned int) * NUM_POPULATIONS));

    {
        const unsigned int numBlocks = oracle ? (startThread / blockSize) : ((paddedPopSize / blockSize) * NUM_POPULATIONS);
        dim3 threads(blockSize, 1);
        dim3 grid(numBlocks, 1);

        CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
        presynapticUpdateIdleThreads<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
        CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
        float time;
        CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
        std::cout << "Idle threads:" << time << std::endl;
    }

    for(unsigned int i = 0; i < NUM_POPULATIONS; i++) {
        deviceToHostCopy(inSyn[i], popSize);

        for(unsigned int j = 0; j < popSize; j++) {
            if(std::fabs(inSyn[i].first[j] - correctInSyn[i][j]) > 0.0001f) {
                std::cerr << "ERROR" << std::endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
