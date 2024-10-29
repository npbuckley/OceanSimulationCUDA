#pragma once
#ifndef UTILS_HEADER
#define UTILS_HEADER

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>

#include <windows.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <vector_types.h>
#include "curand.h"

#include "specs.h"

extern float vbo_time;
extern const int LOGN;

// Complex Number Class
typedef class complex {
public:
	float r;
	float i;

	// Constructors
	__host__ __device__ complex(float _r, float _i) {
		r = _r;
		i = _i;
	}
	__host__ __device__ complex() {
		r = 0.0f;
		i = 0.0f;
	}

	// Complex addition functions
	__host__ __device__ complex& operator+= (const complex& rhs) {
		r += rhs.r;
		i += rhs.i;
		return *this;
	}
	__host__ __device__ complex operator+ (const complex& rhs) {
		return complex(r + rhs.r, i + rhs.i);
	}

	// Complex subtraction
	__host__ __device__ complex& operator-= (const complex& rhs) {
		r -= rhs.r;
		i -= rhs.i;
		return *this;
	}
	__host__ __device__ complex operator- (const complex& rhs) {
		return complex(r - rhs.r, i - rhs.i);
	}

	// Complex multiplication functions
	__host__ __device__ complex& operator*= (const complex& rhs) {
		float temp = r;

		r = r * rhs.r - i * rhs.i;
		i = temp * rhs.i + i * rhs.r;

		return *this;
	}
	__host__ __device__ complex operator* (const complex& rhs) {
		return complex(r * rhs.r - i * rhs.i, r * rhs.i + i * rhs.r);
	}

	// Complex scalar multiplication functions
	__host__ __device__ complex& operator*= (const float& rhs) {
		r *= rhs;
		i *= rhs;
		return *this;
	}
	__host__ __device__ complex operator* (const float& rhs) {
		return complex(r * rhs, i * rhs);
	}

	// Scalar division
	__host__ __device__ complex operator/ (const float& rhs) {
		return complex(r / rhs, i / rhs);
	}

	// Comparison
	__host__ __device__ int operator== (const complex& rhs) {
		return (r == rhs.r) && (i == rhs.i);
	}
	__host__ __device__ int operator!= (const complex& rhs) {
		return !(*this == rhs);
	}
	__host__ __device__ int operator> (const complex& rhs) {
		return (r > rhs.r) && (i > rhs.i);
	}

	// Returns complex conjugate of complex number
	__host__ __device__ complex& conj() {
		i = -i;
		return *this;
	}
};

// Vector2 Class
typedef class vector2 {
public:
	float x;
	float y;

	// Constructors
	__host__ __device__ vector2(float _x, float _y) {
		x = _x;
		y = _y;
	}
	__host__ __device__ vector2() {
		vector2(0.0f, 0.0f);
	}

	// Length of vector
	__host__ __device__ float len() {
		return sqrt(x * x + y * y);
	}

	// Get unit vector
	__host__ __device__ vector2& unit() {
		float length = this->len();

		if (length == 0.0f)
			return *this;

		x /= length;
		y /= length;
		return *this;
	}

	// Vector dot product
	__host__ __device__ float operator* (const vector2& rhs) {
		return x * rhs.x + y * rhs.y;
	}
};

// Vertex type (AoS)
typedef struct {
	complex htilde0;
	complex htild0_conj;
} vertex;

// Butterfly Array
typedef struct {
	complex twiddle;
	int index1;
	int index2;
} butterfly;

void cuda_debug_check();

// Functions and variables shared by the CPU and GPU
namespace shared {
	extern float* d_random_numbers;
	extern float* h_random_numbers;

	void generate_random_numbers();
}

namespace vbo {
	// vbo variables
	extern GLuint vbo;
	extern struct cudaGraphicsResource* cuda_vbo_resource;
	extern void* d_vbo_buffer;

	namespace cpu 
	{
		void createVBO(GLuint* vbo);
		void deleteVBO(GLuint* vbo);
	}

	namespace gpu 
	{
		void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
		void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);
	}
}

#endif