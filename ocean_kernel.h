#pragma once
#ifndef OCEAN_KERNEL_HEADER
#define OCEAN_KERNEL_HEADER

#include "utils.h"
//#include "curand.h"

namespace kernel {
	// htilde0 functions
	__device__ float phillips(int n_prime, int m_prime);
	__global__ void create_htilde_0_kernel(vertex* data, float* rand_nums);

	__device__ float dispersion(int n_prime, int m_prime);
	__device__ complex htilde(int n_prime, int m_prime, float t, complex tilde0, complex tilde0_conj);

	namespace direct_one_step {
		__device__ float3 h_D(vector2 x, float t, vertex* vertices);
		__global__ void calculate_vbo(float3* data, float time, vertex* vertices);
	}

	namespace direct_two_step {
		__global__ void h_D(complex* result, float t, vertex* vertices);
		__global__ void calculate_vbo(float3* data, float time, complex* htilde);
	}

	namespace fft {
		__global__ void reversed(int* result, int LOGN);
		__global__ void butterfly_array(butterfly* result, int* bit_reversed, int LOGN);

		__global__ void h(vertex* htilde0, float time, complex* dy, complex* dxdz);

		__global__ void horizontal_fft(int stage, int pingpong, complex* pingpong0, complex* pingpong1, butterfly* precomuted, int LOGN);
		__global__ void vertical_fft(int stage, int pingpong, complex* pingpong0, complex* pingpong1, butterfly* precomuted, int LOGN);
		__global__ void permute(complex* result);
		__global__ void combine(float3* data, complex* dy, complex* dxdz);
	}

}

#endif