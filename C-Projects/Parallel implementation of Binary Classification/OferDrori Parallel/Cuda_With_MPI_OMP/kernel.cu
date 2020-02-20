
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>



void freeCudaMemory(double* wCuda, int* resultCuda, double *pointsCuda);

cudaError_t calculatePointsCuda(double* points, int N, int K, double * w, int *results);




__global__ void fCuda(double* points, int* N, int* K, double * w, int * results)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int i;
	int sign;
	if (id >= *N)
		return;
	double f = w[*K];
	for (i = 0; i < *K; i++)
	{
		f += points[id * (*K) + i] * w[i];
	}
	if (f >= 0)
		results[id] = 1;
	else
		results[id] = -1;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t calculatePointsCuda(double* points, int N, int K, double * w, int *results)
{
	char errorBuffer[100];
	double* pointsCuda;
	double * wCuda;
	int * resultCuda;
	int* K_cuda;
	int* N_cuda;
	cudaError_t cudaStatus;
	cudaDeviceProp props;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	cudaGetDeviceProperties(&props, 0);

	cudaStatus = cudaMalloc((void**)&K_cuda, sizeof(int));
	cudaStatus = cudaMalloc((void**)&N_cuda, sizeof(int));

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&pointsCuda, sizeof(double)*N*K);
	//checkError(cudaStatus, cudaPoints, "cudaMalloc failed!");
	cudaStatus = cudaMalloc((void**)&wCuda, sizeof(double)*(K+1));
	cudaStatus = cudaMalloc((void**)&resultCuda, sizeof(int) * N);

	//	checkError(cudaStatus, cudaPoints, "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(K_cuda, &K, sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(N_cuda, &N, sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(pointsCuda, points, sizeof(double)*N*K, cudaMemcpyHostToDevice);
	//checkError(cudaStatus, points, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(wCuda, w, sizeof(double)*(K+1), cudaMemcpyHostToDevice);
	
	int numThreads = props.maxThreadsPerBlock < N ? props.maxThreadsPerBlock : N;
	int numBlocks = N / numThreads;
	int extraBlock = N % numThreads != 0;

	// Launch a kernel on the GPU with one thread for each element.
	fCuda <<<numBlocks + extraBlock, numThreads >>>(pointsCuda, N_cuda, K_cuda, wCuda, resultCuda);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	//sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	if (cudaStatus != cudaSuccess)
	{
		printf(errorBuffer);
	}
	//checkError(cudaStatus, dev_c, errorBuffer);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//checkError(cudaStatus, dev_c, errorBuffer);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, resultCuda, N * sizeof(int), cudaMemcpyDeviceToHost);
	//checkError(cudaStatus, dev_c, "cudaMemcpy failed!");

	freeCudaMemory(wCuda, resultCuda, pointsCuda);

	return cudaStatus;
}

void freeCudaMemory(double* wCuda, int* resultCuda, double *pointsCuda)
{
	cudaFree(wCuda);
	cudaFree(resultCuda);
	cudaFree(pointsCuda);
}



