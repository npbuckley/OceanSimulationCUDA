#include "utils.h"

float vbo_time;
const int LOGN = log2(N);

namespace vbo {
    GLuint vbo = 0;
    struct cudaGraphicsResource* cuda_vbo_resource = NULL;
    void* d_vbo_buffer = NULL;

    namespace cpu {
        void createVBO(GLuint* vbo) {
            assert(vbo);

            // create buffer object
            glGenBuffers(1, vbo);
            glBindBuffer(GL_ARRAY_BUFFER, *vbo);

            // initialize buffer object
            glBufferData(GL_ARRAY_BUFFER, N * N * sizeof(float3), 0, GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, 0);

            SDK_CHECK_ERROR_GL();
        }

        void deleteVBO(GLuint* vbo) {
            glBindBuffer(1, *vbo);
            glDeleteBuffers(1, vbo);

            *vbo = 0;
        }
    }

    namespace gpu {
        void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags) {
            assert(vbo);

            // create buffer object
            glGenBuffers(1, vbo);
            glBindBuffer(GL_ARRAY_BUFFER, *vbo);

            // initialize buffer object
            //unsigned int size = MESH_WIDTH * MESH_HEIGHT * 4 * sizeof(float);
            glBufferData(GL_ARRAY_BUFFER, N * N * sizeof(float3), 0, GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, 0);

            // register this buffer object with CUDA
            checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

            SDK_CHECK_ERROR_GL();
        }

        void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {
            // unregister this buffer object with CUDA
            checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

            glBindBuffer(1, *vbo);
            glDeleteBuffers(1, vbo);

            *vbo = 0;
        }
    }
}

void cuda_debug_check() {
    cudaError cerror;
    if (DEBUG) {
        cudaDeviceSynchronize();
        if (cerror = cudaGetLastError()) {
            fprintf(stderr, "error: %s\n", cudaGetErrorString(cerror));
            exit(cerror);
        }
    }
}

float* shared::h_random_numbers;
float* shared::d_random_numbers;

void shared::generate_random_numbers() {
    size_t random_numbers_size = N * N * 4 * sizeof(float);

    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
    cudaMalloc(&d_random_numbers, random_numbers_size);
    curandGenerateNormal(curandGenerator, d_random_numbers, N * N * 4, 0.0, 1.0f);

    switch (MODE) {
    case(CPU):
    case(CORRECTNESS):
        h_random_numbers = (float*)malloc(random_numbers_size);
        cudaMemcpy(h_random_numbers, d_random_numbers, random_numbers_size, cudaMemcpyDeviceToHost);
    }
}