#pragma once
#ifndef SPECS_HEADER
#define SPECS_HEADER

#include "constants.h"

// Options ------------------------------------------------------------------
// Mode Flag (CPU/CPU_PLUS/GPU/CORRECTNESS)
#define MODE		GPU

// Debug Flag (OFF/ON)
// Checks CUDA functions for errors
#define DEBUG		OFF

// VBO Flag (VBO_DIRECT_ONE_STEP/VBO_DIRECT_TWO_STEP/VBO_FFT)
#define VBO_MODE	VBO_FFT

// Number of vertices and waves (Must be multiple of 2)
#define N			2048

// Length of mesh (Must be float)
#define LENGTH		N * 2.0f


// Size of GUI window
#define WINDOW_WIDTH	1080
#define WINDOW_HEIGHT	720

#define REFRESH_DELAY	1 //ms
#define TIMESTEP		100 //ms

// Phillips
#define A				0.8f // Scalar
#define V				40.0f // Wind Speed
#define W				vector2(1.0f, 1.0f) // Wind Direction
#define K_CLAMP	0.00004f // Minimum allowed vector length
#define	L				(V * V / g) // Wind factor
// --------------------------------------------------------------------------

#endif