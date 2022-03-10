#include <stdio.h>
#include "functions.cuh"
#include <math.h>

#define BlockDim 32

template< typename T >
inline __device__ T Inside(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 4 * X[i * N + j] - X[(i + 1) * N + j] - X[(i - 1) * N + j] - X[i * N + j + 1] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T LUp(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 2 * X[i * N + j] - X[(i + 1) * N + j] - X[i * N + j + 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T LLow(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 2 * X[i * N + j] - X[(i - 1) * N + j] - X[i * N + j + 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T RUp(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 2 * X[i * N + j] - X[(i + 1) * N + j] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T RLow(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 2 * X[i * N + j] - X[(i - 1) * N + j] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T Up(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 3 * X[i * N + j] - X[(i + 1) * N + j] - X[i * N + j + 1] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T Low(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 3 * X[i * N + j] - X[(i - 1) * N + j] - X[i * N + j + 1] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T Left(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 3 * X[i * N + j] - X[(i + 1) * N + j] - X[(i - 1) * N + j] - X[i * N + j + 1];
	// value *= denominator;

	return value;
}

template< typename T >
inline __device__ T Right(const int i, const int j, T *X, T *f, const int N, const T h)
{
	// T denominator = 1 / (h * h);
	T value = 3 * X[i * N + j] - X[(i + 1) * N + j] - X[(i - 1) * N + j] - X[i * N + j - 1];
	// value *= denominator;

	return value;
}

template< typename T >
__device__ void ErrorNorm(T *X, T *Y, T *error, const int N)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * N + col;

	if(index != 0)
		return;

	T Error = (T)(0);

	for (int i = 0; i < N * N; ++i)
			Error += (X[i] - Y[i]) * (X[i] - Y[i]);

	Error = sqrt(Error);

	error[0] = Error;
}

template< typename T >
__device__ void SharedMatrixMultiply(T *A, T *B, T *C, const int N)
{
	__shared__ T a[BlockDim * BlockDim], b[BlockDim * BlockDim];

	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int index = row * N + col;

	T sum = 0.0f;

	a[idx] = A[index];
	b[idx] = B[index];

	__syncthreads();

	for (int i = 0; i < BlockDim; i++) 
	{
		sum += a[threadIdx.y * blockDim.x + i] * b[i* blockDim.x + threadIdx.x];
	}

	C[row*N+col] = sum;
}

template< typename T >
__device__ T ReduceTotalError(T *X, const int N);
template< typename T >
__device__ T ReduceTotalError(T *X, const int N)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * N + col;

	if(index != 0)
		return;

	T Error = (T)(0);

	for (int i = 0; i < N * N; ++i)
		Error += X[i] * X[i];

	Error = sqrt(Error);

	printf("|Ax - b| = %lf\n", Error);

	return Error;

}



template< typename T >
__global__ void Solve(T *X, T *Y, T *f, T *error, const int N, const T h, const T eps)
{
	const int I = blockIdx.y*blockDim.y+threadIdx.y;
	const int J = blockIdx.x*blockDim.x+threadIdx.x;
	const int index = I * N + J;

	T (*Aij) (const int, const int, T *, T *, const int, const T);
	// T (*Bij) (T *, const int, const int, T *, T *, const int, const T, const T);

	if(I == 0 && J == 0)
		Aij = LUp<T>;
	else if(I == 0 && J == N-1)
		Aij = RUp<T>;
	else if(I == N-1 && J == 0)
		Aij = LLow<T>;
	else if(I == N-1 && J == N-1)
		Aij = RLow<T>;
	else if((I > 0 && I < N) && J == 0)
		Aij = Left<T>;
	else if((I > 0 && I < N) && J == N-1)
		Aij = Right<T>;
	else if((J > 0 && J < N) && I == 0)
		Aij = Up<T>;
	else if((J > 0 && J < N) && I == N-1)
		Aij = Low<T>;
	else
		Aij = Inside<T>;

	// if(I == J)
	// 	Bij = B_ii<T>;
	// else
	// 	Bij = B_ij<T>;

	const int mask = ((I < N) && (J < N));
	T m = (T)(8) * (T)(N * N) * (T)(cos((M_PI * (double)(h)) / 2.0)), M = (T)(8) * (T)(N * N) * (T)(sin((M_PI * (double)(h)) / 2.0));
	T tau1 = (m + M) / (T)(2) + ((M - m) / (T)(2)) * (T)(cos(M_PI / (2.0 * (double)(N)))), tau2 = (m + M) / (T)(2) + ((M - m) / (T)(2)) * (T)(cos((M_PI * 3.0) / (2.0 * (double)(N))));
	T Xij;

	tau1 = (T)(1) / tau1;
	tau2 = (T)(1) / tau2;

	if(index == 0)
	{
		// printf("sin = %lf\n", sin((M_PI * (double)(h)) / 2.0));
		printf("m = %lf, M = %lf\n", m, M);
		printf("tau1 = %lf, tau2 = %lf\n", tau1, tau2);
	}

	// return;

	// ErrorNorm(X, Y, error, N);
	// if(index == 0)
	// 	printf("error[0] = %lf\n", error[0]);


	// __syncthreads();

	if(mask)
	{
		while(error[0] > eps)
		{
			// if(index == 0)
			// 	printf("%.20lf = -%.20lf * %.20lf + %.20lf * %.20lf * %.20lf\n", Y[I * N + J], Aij(I, J, X, f, N, h), tau1, tau1, f[I * N + J], h * h);
			Xij = X[I * N + J];
			T val = -Aij(I, J, X, f, N, h) * tau1 + tau1 * f[I * N + J] * h * h;
			Y[I * N + J] = val;
			Y[I * N + J] += Xij;
			// if(index == 0)
			// 	printf("%.20lf = %.20lf -> ", Y[I * N + J], val);
			__syncthreads();

			ErrorNorm(X, Y, error, N);

			if(index == 0)
				printf("error[0] = %.20lf\n", error[0]);

			__syncthreads();

			if(error[0] < eps)
			{
				X[I * N + J] = Y[I * N + J];
				break;
			}

			X[I * N + J] = Y[I * N + J];
			__syncthreads();

			Xij = X[I * N + J];
			// if(index == 0)
			// 	printf("%.20lf = -%.20lf * %.20lf + %.20lf * %.20lf * %.20lf\n", Y[I * N + J], Aij(I, J, X, f, N, h), tau2, tau2, f[I * N + J], h * h);
			val = -Aij(I, J, X, f, N, h) * tau2 + tau2 * f[I * N + J] * h * h;
			Y[I * N + J] = val;
			Y[I * N + J] += Xij;
			// if(index == 0)
			// 	printf("%.20lf = %.20lf\n", Y[I * N + J], val);
			__syncthreads();

			ErrorNorm(X, Y, error, N);
			if(index == 0)
				printf("error[0] = %.20lf\n", error[0]);

			__syncthreads();

			X[I * N + J] = Y[I * N + J]; 
			__syncthreads();

		}
	}

	if(mask)
	{
		Y[I * N + J] = Aij(I, J, X, f, N, h) - f[I * N + J] * h * h;
		__syncthreads();

		ReduceTotalError(Y, N);
		
		// printf("error = %lf\n", Aij(I, J, X, f, N, h) - f[I * N + J] * h * h);
	}

}
