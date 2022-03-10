#pragma once

/**
 * Inside if point is inside Omega   		
 * Up if 0-row not corner										
 * Low if N-1-row not corner
 * Left if 0-column not corner	
 * Right if N-1-column not corner
 * LUp if corner							
 * LLow if corner							
 * RUp if corner							
 * RLow if corner							
 */

template< typename T >
inline __device__ T Inside(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T LUp(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T LLow(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T RUp(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T RLow(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T Up(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T Low(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T Left(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
inline __device__ T Right(const int i, const int j, T *X, T *f, const int N, const T h);

template< typename T >
__device__ void ErrorNorm(T *X, T *Y, T *error, const int N);

template< typename T >
__device__ void SharedMatrixMultiply(T *A, T *B, T *C, const int N);

template< typename T >
__device__ T B_ii(T *A, const int i, const int j, T *X, T *f, const int N, const T h, const T tau);

template< typename T >
__device__ T B_ij(T *A, const int i, const int j, T *X, T *f, const int N, const T h, const T tau);

template< typename T >
__global__ void Solve(T *X, T *Y, T *f, T *error, const int N, const T h, const T eps);

