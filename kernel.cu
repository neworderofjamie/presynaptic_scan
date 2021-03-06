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
#define BLOCK_IDX_COUNT 8272
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
unsigned int oracleMergedGroupStartID[NUM_POPULATIONS];
unsigned int overallocateMergedGroupStartID[NUM_POPULATIONS];
unsigned int mergedGroupBlockIdx[BLOCK_IDX_COUNT];
MergedPresynapticUpdateGroup mergedGroups[NUM_POPULATIONS];

// Device globals
__device__ unsigned int d_max;
__device__ unsigned int d_mergedGroupStartID[NUM_POPULATIONS];
__device__ __constant__ MergedPresynapticUpdateGroup d_mergedGroups[NUM_POPULATIONS];
__device__ __constant__ unsigned int d_mergedGroupBlockIdx[BLOCK_IDX_COUNT];

// Presynaptic update kernel which assumes merged group start ids cover entire grid
__global__ void presynapticUpdateBlockIdx()
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    const unsigned int groupID = d_mergedGroupBlockIdx[blockIdx.x];
    struct MergedPresynapticUpdateGroup *group = &d_mergedGroups[groupID];
    const unsigned int groupStartID = d_mergedGroupStartID[groupID];
    const unsigned int lid = id - groupStartID;

    if(lid < group->srcSpkCnt[0]) {
        const unsigned int preInd = group->srcSpk[lid];
        atomicAdd(&group->inSyn[preInd], group->weight[preInd]);
    }
}


// Presynaptic update kernel which assumes merged group start ids cover entire grid
__global__ void presynapticUpdate()
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

// Presynaptic update kernel which reads max overall threads from global
 __global__ void presynapticUpdateMax()
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < d_max) {
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
}

template<unsigned int B, unsigned P, bool dynamic>
__global__ void treeScan()
{
    __shared__ unsigned int shIntermediate[B];

    // Copy padded spike counts into shared memory
    if(threadIdx.x < NUM_POPULATIONS) {
        shIntermediate[threadIdx.x] = ((d_mergedGroups[threadIdx.x].srcSpkCnt[0] + P - 1) / P) * P;  
    }
    
    // Perform tree scan
    for(unsigned int d = 1; d < B; d<<=1) {
        __syncthreads();
        const float temp = (threadIdx.x >= d) ? shIntermediate[threadIdx.x - d] : 0;
        __syncthreads();
        shIntermediate[threadIdx.x] += temp;
    }

    // Insert zero in first entry
    if(threadIdx.x == 0) {
        d_mergedGroupStartID[0] = 0;
    }

    // Copy in shared memory
    if(threadIdx.x < (NUM_POPULATIONS - 1)) {       
        d_mergedGroupStartID[threadIdx.x + 1] = shIntermediate[threadIdx.x];
    }
    
    // If this is the last thread, launch presynaptic update kernel
    if(threadIdx.x == (NUM_POPULATIONS - 1)) {
        if(dynamic) {
            dim3 threads(P, 1);
            dim3 grid(shIntermediate[threadIdx.x] / P, 1);
            presynapticUpdate<<<grid, threads>>>();
        }
        else {
            d_max = shIntermediate[threadIdx.x];
        }
    }
}

template<unsigned int B, unsigned P, bool dynamic>
__global__ void treeScanWarpShuffle()
{
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    constexpr unsigned int numWarps = B / 32;

    __shared__ unsigned int shIntermediate[numWarps];

    // Read and pad number of spikes
    unsigned int paddedNumSpikes = (threadIdx.x < NUM_POPULATIONS) ? ((d_mergedGroups[threadIdx.x].srcSpkCnt[0] + P - 1) / P) * P : 0;
    
    // Perform warp scan
    for(unsigned int d = 1; d < 32; d <<= 1) {
        const unsigned int temp = __shfl_up_sync(0xFFFFFFFF, paddedNumSpikes, d);
        if(lane >= d) {
            paddedNumSpikes += temp;
        }
    }
    
    // Copy warp scans into shared memory
    if(lane == 31) {
        shIntermediate[warp] = paddedNumSpikes;
    }
    __syncthreads();

    // If this is the first warp
    if(warp == 0) {
        // Read warp scan
        unsigned int warpScan = (threadIdx.x < numWarps) ? shIntermediate[threadIdx.x] : 0;
         
        for (unsigned int d = 1; d < numWarps; d <<= 1) {
            const unsigned int temp = __shfl_up_sync(0xFFFFFFFF, warpScan, d);
            if(lane >= d) {
                warpScan += temp;
            }
        }
        if(threadIdx.x < numWarps) {
            shIntermediate[threadIdx.x] = warpScan;
        }
    }

    __syncthreads();
    if(warp > 0) {
        paddedNumSpikes += shIntermediate[warp - 1];
    }

    // Insert zero in first entry
    if(threadIdx.x == 0) {
        d_mergedGroupStartID[0] = 0;
    }

    // Copy in shared memory
    if(threadIdx.x < (NUM_POPULATIONS - 1)) {       
        d_mergedGroupStartID[threadIdx.x + 1] = paddedNumSpikes;
    }
    
    // If this is the last thread, launch presynaptic update kernel
    if(threadIdx.x == (NUM_POPULATIONS - 1)) {
        if(dynamic) {
            dim3 threads(P, 1);
            dim3 grid(paddedNumSpikes / P, 1);
            presynapticUpdate<<<grid, threads>>>();
        }
        else {
            d_max = paddedNumSpikes;
        }
    }
}

template<unsigned int B>
__global__ void matrixScanSM()
{
    // **TODO** prevent bank conflicts
    __shared__ unsigned int shMatrix[B][B];
    __shared__ unsigned int shIntermediate[B];

    // 1) Row reduce
    // Loop through columns
    // **NOTE** because spike counts are accessed indirectly, no particular point in an additional stage to try and read coalesced
    {
        unsigned int sum = 0;
        unsigned int idx = threadIdx.x * B;
        for(unsigned int j = 0; j < B; j++) {
            // If there is a population here
            if(idx < NUM_POPULATIONS) {
                // Read spike count, write to shared memory array and add to sum
                const unsigned int spikeCount = d_mergedGroups[idx].srcSpkCnt[0];
                shMatrix[threadIdx.x][j] = spikeCount;
                sum += spikeCount;
            }
            idx++;
        }

        printf("Row sum %u = %u\n", threadIdx.x, sum);
        // Copy sum to shared memory array
        shIntermediate[threadIdx.x] = sum;
    }
    __syncthreads();

    // 2) Column scan
    // **TODO** parallel scan
    {
        if(threadIdx.x == 0) {
            unsigned int scan = 0;
            for(unsigned int j = 0; j < B; j++) {
                const unsigned int element = shIntermediate[j];
                shIntermediate[j] = scan;
                scan += element;
            }
        }
    }
    __syncthreads();
    // 3) Row scan
    {
        unsigned int scan = shIntermediate[threadIdx.x];
        unsigned int idx = threadIdx.x * B;
        for(unsigned int j = 0; j < B; j++) {
            // If there is a population here
            if(idx < NUM_POPULATIONS) {
                // Write start ID back to global memory
                // **NOTE** HERE coalescing would be worthwhile
                d_mergedGroupStartID[idx] = scan;

                // Add this matrix element to scan
                scan += shMatrix[threadIdx.x][j];
            }
            idx++;
        }
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
void checkOutput(const std::vector<float> (&correctInSyn)[NUM_POPULATIONS],
                 HostDeviceArray<float> (&inSyn)[NUM_POPULATIONS], const unsigned int (&popSizes)[NUM_POPULATIONS])
{
     for(unsigned int i = 0; i < NUM_POPULATIONS; i++) {
        deviceToHostCopy(inSyn[i], popSizes[i]);

        for(unsigned int j = 0; j < popSizes[i]; j++) {
            if(std::fabs(inSyn[i].first[j] - correctInSyn[i][j]) > 0.0001f) {
                std::cerr << "\tFailed" << std::endl;
                return;
            }
        }
    }
}
//----------------------------------------------------------------------------
void zeroISyn(HostDeviceArray<float> (&inSyn)[NUM_POPULATIONS], const unsigned int (&popSizes)[NUM_POPULATIONS])
{
    for(unsigned int i = 0; i < NUM_POPULATIONS; i++) {
        // Zero host inSyn
        std::fill_n(&inSyn[i].first[0], popSizes[i], 0.0f);

        // Copy to device
        hostToDeviceCopy(inSyn[i], popSizes[i]);
    }
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Population sizes and number of spikes
    constexpr unsigned int popSizes[NUM_POPULATIONS] = {3000, 
                                                        60000, 60000, 
                                                        30000, 30000, 
                                                        15000, 15000, 15000, 
                                                        7500, 7500, 7500,
                                                        2000, 2000, 2000,
                                                        4000, 4000,
                                                        10};
    constexpr unsigned int numSpikes[NUM_POPULATIONS] = {500, 9600, 3700, 2600, 1200, 990, 770, 600, 
                                                         300, 790, 670, 260, 430, 230, 560, 750, 5};
    /*unsigned int popSizes[NUM_POPULATIONS];
    unsigned int numSpikes[NUM_POPULATIONS];
    std::fill(std::begin(popSizes), std::end(popSizes), 50000);
    std::fill(std::begin(numSpikes), std::end(numSpikes), 5000);*/
    
    
    constexpr unsigned int blockSize = 32;
    
    try
    {
        // Pad population sizes
        unsigned int paddedPopSizes[NUM_POPULATIONS];
        std::transform(std::begin(popSizes), std::end(popSizes), std::begin(paddedPopSizes),
                       [blockSize](unsigned int popSize){ return ((popSize + blockSize - 1) / blockSize) * blockSize; });

        constexpr unsigned int paddedGroupSize = ((NUM_POPULATIONS + 32 - 1) / 32) * 32;
        std::cout << "Padded group size:" << paddedGroupSize << std::endl;

        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        cudaEvent_t updateStart;
        cudaEvent_t updateEnd;
        CHECK_CUDA_ERRORS(cudaEventCreate(&updateStart));
        CHECK_CUDA_ERRORS(cudaEventCreate(&updateEnd));

        std::mt19937 rng;
        std::normal_distribution<float> weightDist(0.0f, 0.25f);

        HostDeviceArray<float> inSyn[NUM_POPULATIONS];
        std::vector<float> correctInSyn[NUM_POPULATIONS];

        unsigned int startThread = 0;
        unsigned int overallocatedStartThread = 0;
        for(unsigned int i = 0; i < NUM_POPULATIONS; i++) {
            // Resize and zero correct insyn vector
            correctInSyn[i].resize(popSizes[i], 0.0f);

            // Allocate memory
            inSyn[i] = allocateHostDevice<float>(popSizes[i]);
            auto srcSpkCnt = allocateHostDevice<unsigned int>(1);
            auto srcSpk = allocateHostDevice<unsigned int>(popSizes[i]);
            auto weight = allocateHostDevice<float>(popSizes[i]);

            // Create distribution of spike IDs
            std::uniform_int_distribution<unsigned int> spikeDist(0, popSizes[i] - 1);

            // Generate random spikes
            srcSpkCnt.first[0] = numSpikes[i];
            std::generate_n(&srcSpk.first[0], numSpikes[i], [&rng, &spikeDist]() { return spikeDist(rng); });

            // Generate weights
            std::generate_n(&weight.first[0], popSizes[i], [&rng, &weightDist]() { return weightDist(rng); });

            // Calculate correct output
            for(unsigned int j = 0; j < numSpikes[i]; j++) {
                const unsigned int ind = srcSpk.first[j];
                correctInSyn[i][ind] += weight.first[ind];
            }

            // Upload
            hostToDeviceCopy(srcSpkCnt, 1, true);
            hostToDeviceCopy(srcSpk, popSizes[i], true);
            hostToDeviceCopy(weight, popSizes[i], true);

            // Build struct with device pointers
            mergedGroups[i].inSyn = inSyn[i].second;
            mergedGroups[i].srcSpk = srcSpk.second;
            mergedGroups[i].srcSpkCnt = srcSpkCnt.second;
            mergedGroups[i].weight = weight.second;

            // Populate block IDs
            std::fill(&mergedGroupBlockIdx[overallocatedStartThread / blockSize], &mergedGroupBlockIdx[(overallocatedStartThread + paddedPopSizes[i]) / blockSize], i);

            // Calculate static start ID
            oracleMergedGroupStartID[i] = startThread;
            overallocateMergedGroupStartID[i] = overallocatedStartThread;

            // Sum padded spikes
            startThread += ((numSpikes[i] + blockSize - 1) / blockSize) * blockSize;
            overallocatedStartThread += paddedPopSizes[i];
        }
        std::cout << "Optimal number of threads:" << startThread << ", Overallocated number of threads:" << overallocatedStartThread << std::endl;
        assert(BLOCK_IDX_COUNT == (overallocatedStartThread / blockSize));
        

        // Copy merged group structures to symbols
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_mergedGroups, &mergedGroups[0], sizeof(MergedPresynapticUpdateGroup) * NUM_POPULATIONS));
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_mergedGroupBlockIdx, &mergedGroupBlockIdx[0], sizeof(unsigned int) * BLOCK_IDX_COUNT));
        
        // Naive tree-scan with host overallocated kernel launch
        {
            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            const unsigned int numBlocks = overallocatedStartThread / blockSize;
            dim3 presynapticThreads(blockSize, 1);
            dim3 presynapticGrid(numBlocks, 1);
            dim3 threads(paddedGroupSize, 1);
            dim3 grid(1, 1);
            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            treeScan<paddedGroupSize, blockSize, false> << <grid, threads >> > ();
            presynapticUpdateMax << <presynapticGrid, presynapticThreads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Tree scan host overallocated kernel:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }

        // Warp-shuffle based tree-scan with host overallocated kernel launch
        {
            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            const unsigned int numBlocks = overallocatedStartThread / blockSize;
            dim3 presynapticThreads(blockSize, 1);
            dim3 presynapticGrid(numBlocks, 1);

            dim3 threads(paddedGroupSize, 1);
            dim3 grid(1, 1);
            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            treeScanWarpShuffle<paddedGroupSize, blockSize, false> << <grid, threads >> > ();
            presynapticUpdateMax << <presynapticGrid, presynapticThreads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Tree scan warp shuffle overallocated kernel:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }

        // Naive tree-scan using dynamic parallelism
        {
            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            dim3 threads(paddedGroupSize, 1);
            dim3 grid(1, 1);
            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            treeScan<paddedGroupSize, blockSize, true> << <grid, threads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Tree scan dynamic parallelism:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }

        // Warp-shuffle based tree-scan using dynamic parallelism
        {
            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            dim3 threads(paddedGroupSize, 1);
            dim3 grid(1, 1);
            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            treeScanWarpShuffle<paddedGroupSize, blockSize, true> << <grid, threads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Tree scan warp shuffle dynamic parallelism:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }

        // Oracle version with perfectly sized groups and kernel
        {
            // Copy perfect 'oracle' group start IDs
            CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_mergedGroupStartID, &oracleMergedGroupStartID[0], sizeof(unsigned int) * NUM_POPULATIONS));

            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            const unsigned int numBlocks = startThread / blockSize;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            presynapticUpdate << <grid, threads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Oracle:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }

        // Overallocated version
        {
            // Copy overallocated group start IDs
            CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_mergedGroupStartID, &overallocateMergedGroupStartID[0], sizeof(unsigned int) * NUM_POPULATIONS));

            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            const unsigned int numBlocks = overallocatedStartThread / blockSize;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            presynapticUpdate << <grid, threads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Overallocated:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }
        
        // Update block IDX version
        {
            // Copy overallocated group start IDs
            CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_mergedGroupStartID, &overallocateMergedGroupStartID[0], sizeof(unsigned int) * NUM_POPULATIONS));

            // Zero ISyn
            zeroISyn(inSyn, popSizes);

            const unsigned int numBlocks = overallocatedStartThread / blockSize;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            CHECK_CUDA_ERRORS(cudaEventRecord(updateStart));
            presynapticUpdateBlockIdx << <grid, threads >> > ();
            CHECK_CUDA_ERRORS(cudaEventRecord(updateEnd));
            CHECK_CUDA_ERRORS(cudaEventSynchronize(updateEnd));
            float time;
            CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, updateStart, updateEnd));
            std::cout << "Block IDX:" << time << std::endl;
            checkOutput(correctInSyn, inSyn, popSizes);
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
