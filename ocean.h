#pragma once
#ifndef OCEAN_HEADER
#define OCEAN_HEADER

#include "utils.h"
#include "ocean_kernel.h"

//extern complex htildeM[N * N];
extern complex* htildeM;

extern complex* d_dy;
//extern complex Dy[N * N];
extern complex* Dy;

namespace ocean
{
	namespace cpu
	{
		float phillips(int n_prime, int m_prime);
		vertex* create_htilde_0();

		float dispersion(int n_prime, int m_prime);
		complex htilde(int n_prime, int m_prime, float t);
		float4 h(vector2 x, float t);

		namespace direct {
			float3* calculate_vbo(GLuint* vbo, float time);
		}

		namespace fft {
			int reversed(int i);
			butterfly* calculate_butterfly();
			complex* h(float t);
			complex* horizontal_fft(int* pingpong, complex* Buffer0, int logSize);
			complex* vertical_fft(int* pingpong, complex* Buffer0, int logSize);
			complex* permute(complex* result);
			float3* calculate_vbo(GLuint* vbo, float t);
		}
	}

	namespace gpu
	{
		vertex* create_htilde_0();

		namespace direct {
			namespace one_step {
				void calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time);
			}
			namespace two_step {
				void h(float t);
				void calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time);
			}
		}

		namespace fft {
			int* calculate_reverse();
			butterfly* calculate_butterfly();
			complex* h(float t);
			complex* horizontal_fft(int* pingpong, complex* input, complex* pingpongarr, cudaStream_t stream);
			complex* vertical_fft(int* pingpong, complex* input, complex* pingpongarr, cudaStream_t stream);
			complex* permute(complex* input, cudaStream_t stream);
			float3* calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time);
		}
	}
}

#endif