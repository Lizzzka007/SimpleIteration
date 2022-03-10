#include <stdio.h>
#include <iostream>
#include "functions.cuh"
#include "functions.cu"
#include <math.h>

using namespace std;

/**
 * Omega := [a, b]^2 
 */

#define a 0.0
#define b 1.0
#define BlockDim 32

typedef float Type;

template< typename T >
inline T func(const T x, const T y);
template< typename T >
inline T func(const T x, const T y)
{
	// return -x * x + y;
	return cos(M_PI * x) * cos(M_PI * y * (T)(3));
}

template< typename T >
void InitData(T *X, T *Y, T *f, const int N);
template< typename T >
void InitData(T *X, T *Y, T *f, const int N)
{
	T x, y;
	float local_a = a, local_b = b;

	for (int i = 0; i < N; ++i)
	{
		x = ((local_b - local_a) / ((float)(N))) * (float)(i);

		for (int j = 0; j < N; ++j)
		{
			y = ((local_b - local_a) / ((float)(N))) * (float)(j);

			X[i * N + j] = 1.0;
			Y[i * N + j] = (T)(i);
			f[i * N + j] = func((T)(x), (T)(y));
		}
	}
}

template< typename T >
void PrintRes(T *X, const int N);
template< typename T >
void PrintRes(T *X, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		printf("%lf ", X[i]);
	}

	printf("\n");
}

int main(void)
{
	int N;
	float GPUtime;
	Type h, eps;
	Type *host_X_i, *host_X_ip, *host_f, *host_error;
	Type *dev_X_i, *dev_X_ip, *dev_f, *dev_error;
	cudaEvent_t start, stop;

	printf("Enter N: \n");
	cin >> N;
	printf("Enter accuracy: \n");
	cin >> eps;

	dim3 Blocks(ceil((N * N) / (float)(BlockDim * BlockDim)), 1, 1);           
	dim3 Threads(BlockDim, BlockDim, 1 );

	h = ((Type)(b - a)) / ((Type)(N));

	host_X_i = (Type*) malloc(sizeof(Type) * N * N );
	host_X_ip = (Type*) malloc(sizeof(Type) * N * N );
	host_error = (Type*) malloc(sizeof(Type) * 1 );
	host_f = (Type*) malloc(sizeof(Type) * N * N );

	host_error[0] = (Type)(1000);

	cudaMalloc(&dev_X_i, sizeof(Type) * N * N );
	cudaMalloc(&dev_X_ip, sizeof(Type) * N * N );
	cudaMalloc(&dev_error, sizeof(Type) * 1 );
	cudaMalloc(&dev_f, sizeof(Type) * N * N );

	InitData(host_X_i, host_X_ip, host_f, N);

	// PrintRes(host_f, N * N);

	// return;

	cudaMemcpy(dev_X_i, host_X_i, sizeof(Type) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_f, host_f, sizeof(Type) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_X_ip, host_X_ip, sizeof(Type) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_error, host_error, sizeof(Type) * 1, cudaMemcpyHostToDevice);

	cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
	cudaEventRecord ( start, 0 );

	Solve<<<Blocks, Threads>>> (dev_X_i, dev_X_ip, dev_f, dev_error, N, h, eps); 

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(host_X_i, dev_X_i, sizeof(Type) * N * N, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&GPUtime, start, stop);

    GPUtime *= 0.001;

    printf("Elapsed time is %10.3e\n\n", GPUtime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    PrintRes(host_X_i, N * N);

	free(host_X_i);
	free(host_X_ip);
	free(host_error);
	free(host_f);

	cudaFree(dev_X_i);
	cudaFree(dev_X_ip);
	cudaFree(dev_error);
	cudaFree(dev_f);

	return 0;
}