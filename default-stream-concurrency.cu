#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Utilities.cuh"

using namespace std;

#define NUM_THREADS 32
#define NUM_BLOCKS 16
#define NUM_STREAMS 3

__global__ void kernel(const int *in, int *out, int N)
{
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int end =  N;
    for (int i = start; i < end; i += blockDim.x * gridDim.x)
    {
        out[i] = in[i] * in[i];
    }
}

int main()
{
    const int N = 6000000;
    
    // --- Host side input data allocation and initialization. Registering host memory as page-locked (required for asynch cudaMemcpyAsync).
    int *h_in = new int[N];	for(int i = 0; i < N; i++) h_in[i] = 5;
    gpuErrchk(cudaHostRegister(h_in, N * sizeof(int), cudaHostRegisterPortable));

    // --- Host side input data allocation and initialization. Registering host memory as page-locked (required for asynch cudaMemcpyAsync).
    int *h_out = new int[N]; for(int i = 0; i < N; i++) h_out[i] = 0;
	gpuErrchk(cudaHostRegister(h_out, N * sizeof(int), cudaHostRegisterPortable));

    // --- Host side check results vector allocation and initialization
    int *h_checkResults = new int[N]; for(int i = 0; i < N; i++) h_checkResults[i] = h_in[i] * h_in[i];

    // --- Device side input data allocation.
    int *d_in = 0;				gpuErrchk(cudaMalloc((void **)&d_in, N * sizeof(int)));

    // --- Device side output data allocation. 
	int *d_out = 0;				gpuErrchk( cudaMalloc((void **)&d_out, N * sizeof(int)));
    
	int streamSize = N / NUM_STREAMS;
    size_t streamMemSize = N * sizeof(int) / NUM_STREAMS;

	// --- Set kernel launch configuration
    dim3 nThreads		= dim3(NUM_THREADS,1,1);
    dim3 nBlocks		= dim3(NUM_BLOCKS, 1,1);
    dim3 subKernelBlock = dim3((int)ceil((float)nBlocks.x / 2));

    // --- Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++)
        gpuErrchk(cudaStreamCreate(&streams[i]));

	/**************************/
	/* BREADTH-FIRST APPROACH */
	/**************************/
	
	int offset = 0;
	cudaMemcpyAsync(&d_in[offset], &h_in[offset], streamMemSize, cudaMemcpyHostToDevice,     0);
	for(int i = 1; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_in[offset], &h_in[offset], streamMemSize, cudaMemcpyHostToDevice,     streams[i]);
	}

    kernel<<<subKernelBlock, nThreads>>>(&d_in[offset], &d_out[offset],   streamSize/2);
    kernel<<<subKernelBlock, nThreads>>>(&d_in[offset + streamSize/2],    &d_out[offset +  streamSize/2], streamSize/2);

	for(int i = 1; i < NUM_STREAMS; i++)
    {
        int offset = i * streamSize;
        kernel<<<subKernelBlock, nThreads, 0, streams[i]>>>(&d_in[offset], &d_out[offset],   streamSize/2);
        kernel<<<subKernelBlock, nThreads, 0, streams[i]>>>(&d_in[offset + streamSize/2],    &d_out[offset +  streamSize/2], streamSize/2);
    }

	for(int i = 1; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&h_out[offset], &d_out[offset], streamMemSize, cudaMemcpyDeviceToHost,   streams[i]);
	}

	cudaMemcpyAsync(&h_out[offset], &d_out[offset], streamMemSize, cudaMemcpyDeviceToHost,   0);
	for(int i = 1; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&h_out[offset], &d_out[offset], streamMemSize, cudaMemcpyDeviceToHost,   0);
	}

	for(int i = 0; i < NUM_STREAMS; i++)
        gpuErrchk(cudaStreamSynchronize(streams[i]));

    gpuErrchk(cudaDeviceSynchronize());

    // --- Release resources
    gpuErrchk(cudaHostUnregister(h_in));
    gpuErrchk(cudaHostUnregister(h_out));
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));

    for(int i = 0; i < NUM_STREAMS; i++)
        gpuErrchk(cudaStreamDestroy(streams[i]));

    cudaDeviceReset();  

    // --- GPU output check
    int sum = 0;
    for(int i = 0; i < N; i++)      
        sum += h_checkResults[i] - h_out[i];

    cout << "Error between CPU and GPU: " << sum << endl;

    delete[] h_in;
    delete[] h_out;
    delete[] h_checkResults;

    return 0;
}
