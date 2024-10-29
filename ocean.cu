#include "ocean.h"
#include "ocean_kernel.h"

vertex* d_vertices;
float* d_rand_nums;

//For debugging
vertex* h_vertices;
butterfly* h_butterfly;
int* h_reverse;
complex* h_dy;
complex* h_dxdz;
float3* h_data;

// For two step
complex* d_htilde;

// For fft
complex* d_dy;
complex* d_dxdz;
int* d_reverse;
butterfly* d_butterfly_pre;
complex* d_pingpong1;
complex* d_pingpong2;

cudaEvent_t start, stop;

vertex* ocean::gpu::create_htilde_0() {
    // Create device vertex array
    size_t vertices_size = N * N * sizeof(vertex);
    checkCudaErrors(cudaMalloc(&d_vertices, vertices_size));

    // Allocate height array if doing two step
    if (VBO_MODE == VBO_DIRECT_TWO_STEP || MODE == CORRECTNESS) {
        checkCudaErrors(cudaMalloc(&d_htilde, N * N * sizeof(complex)));
    }

    // Allocate partial arrays and pingpong array of FFT
    if (VBO_MODE == VBO_FFT || MODE == CORRECTNESS) {
        checkCudaErrors(cudaMalloc(&d_dy, N * N * sizeof(complex)));
        checkCudaErrors(cudaMalloc(&d_dxdz, N * N * sizeof(complex)));
        checkCudaErrors(cudaMalloc(&d_pingpong1, N * N * sizeof(complex)));
        checkCudaErrors(cudaMalloc(&d_pingpong2, N * N * sizeof(complex)));
    }

    // Create vbo timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute the startup
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);
    kernel::create_htilde_0_kernel << < grid, block >> > (d_vertices, shared::d_random_numbers);
    cuda_debug_check();


    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        h_vertices = (vertex*)malloc(vertices_size);
        checkCudaErrors(cudaMemcpy(h_vertices, d_vertices, vertices_size, cudaMemcpyDeviceToHost));
        return h_vertices;
    }
    return SUCCESS;
}


void ocean::gpu::direct::one_step::calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time) {
    float3* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));

    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));

    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    // Call Kernel
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    kernel::direct_one_step::calculate_vbo << < grid, block >> > (dptr, time, d_vertices);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    // Set vbo time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&vbo_time, start, stop);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}


void ocean::gpu::direct::two_step::h(float t) {
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    // Create height dependent array
    kernel::direct_two_step::h_D << < grid, block >> > (d_htilde, t, d_vertices);

    cuda_debug_check();
}

void ocean::gpu::direct::two_step::calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time) {
    float3* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));

    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));

    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Create height dependent array
    h(time);
    //kernel::direct_two_step::h_D << < grid, block >> > (d_htilde, time, d_vertices);

    //cuda_debug_check();

    // Create VBO
    kernel::direct_two_step::calculate_vbo << < grid, block >> > (dptr, time, d_htilde);

    cuda_debug_check();

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    // Set vbo time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&vbo_time, start, stop);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}


int* ocean::gpu::fft::calculate_reverse() {
    int reverse_size = N * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_reverse, reverse_size));

    // Create Reversed
    dim3 block(min(N, 32), 1, 1);
    dim3 grid(ceil(N / (float)block.x), 1, 1);
    kernel::fft::reversed << < grid, block >> > (d_reverse, LOGN);
    cuda_debug_check();

    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        h_reverse = (int*)malloc(reverse_size);
        checkCudaErrors(cudaMemcpy(h_reverse, d_reverse, reverse_size, cudaMemcpyDeviceToHost));
        return h_reverse;
    }

    return SUCCESS;
}

butterfly* ocean::gpu::fft::calculate_butterfly() {
    // Allocate butterfly matrix
    int butterfly_size = LOGN * N * sizeof(butterfly);
    checkCudaErrors(cudaMalloc(&d_butterfly_pre, butterfly_size));

    // Call butterfly matrix kernel
    dim3 block(min(LOGN, 32), min(N, 32), 1);
    dim3 grid(ceil(LOGN / (float)block.x), ceil(N / (float)block.y), 1);
    kernel::fft::butterfly_array << < grid, block >> > (d_butterfly_pre, d_reverse, LOGN);
    cuda_debug_check();

    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        h_butterfly = (butterfly*)malloc(butterfly_size);
        checkCudaErrors(cudaMemcpy(h_butterfly, d_butterfly_pre, butterfly_size, cudaMemcpyDeviceToHost));
        return h_butterfly;
    }
    return SUCCESS;
}

complex* ocean::gpu::fft::h(float t) {
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);
    kernel::fft::h << < grid, block >> > (d_vertices, t, d_dy, d_dxdz);
    cuda_debug_check();

    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        int sizes = N * N * sizeof(complex);
        h_dy = (complex*)malloc(sizes);
        h_dxdz = (complex*)malloc(sizes);
        checkCudaErrors(cudaMemcpy(h_dy, d_dy, sizes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_dxdz, d_dxdz, sizes, cudaMemcpyDeviceToHost));
        return h_dy;
    }
    return SUCCESS;
}

complex* ocean::gpu::fft::horizontal_fft(int* pingpong, complex* input, complex* pingpongarr, cudaStream_t stream) {
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    for (int x = 0; x < LOGN; ++x) {
        *pingpong ^= 1;
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        kernel::fft::horizontal_fft << < grid, block, 0, stream >> > (x, *pingpong, input, pingpongarr, d_butterfly_pre, LOGN);
        cuda_debug_check();
    }

    if (MODE == CORRECTNESS) {
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        int sizes = N * N * sizeof(complex);
        h_dy = (complex*)malloc(sizes);
        checkCudaErrors(cudaMemcpy(h_dy, input, sizes, cudaMemcpyDeviceToHost));
        return h_dy;
    }
    return SUCCESS;
}

complex* ocean::gpu::fft::vertical_fft(int* pingpong, complex* input, complex* pingpongarr, cudaStream_t stream) {
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    for (int x = 0; x < LOGN; ++x) {
        *pingpong ^= 1;
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        kernel::fft::vertical_fft << < grid, block, 0, stream >> > (x, *pingpong, input, pingpongarr, d_butterfly_pre, LOGN);
        cuda_debug_check();
    }

    if (MODE == CORRECTNESS) {
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        int sizes = N * N * sizeof(complex);
        h_dy = (complex*)malloc(sizes);
        checkCudaErrors(cudaMemcpy(h_dy, input, sizes, cudaMemcpyDeviceToHost));
        return h_dy;
    }
    return SUCCESS;
}

complex* ocean::gpu::fft::permute(complex* input, cudaStream_t stream) {
    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    cudaStreamSynchronize(stream);
    kernel::fft::permute << < grid, block, 0, stream >> > (input);
    cuda_debug_check();

    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        int sizes = N * N * sizeof(complex);
        h_dy = (complex*)malloc(sizes);
        checkCudaErrors(cudaMemcpy(h_dy, input, sizes, cudaMemcpyDeviceToHost));
        return h_dy;
    }
    return SUCCESS;
}

float3* ocean::gpu::fft::calculate_vbo(struct cudaGraphicsResource** vbo_resource, float time) {
    float3* dptr;
    if (vbo_resource != NULL)
        checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));

    size_t num_bytes;
    if (vbo_resource != NULL)
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));
    else
        cudaMalloc(&dptr, N * N * sizeof(float3));

    dim3 block(min(N, 32), min(N, 32), 1);
    dim3 grid(ceil(N / (float)block.x), ceil(N / (float)block.y), 1);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Create partial displacement
    h(time);

    // Horizontal Displacement FFTs
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int pingpong = 0;
    int pingpong2 = 0;
    horizontal_fft(&pingpong, d_dxdz, d_pingpong1, stream1);
    horizontal_fft(&pingpong2, d_dy, d_pingpong2, stream2);
    vertical_fft(&pingpong, d_dxdz, d_pingpong1, stream1);
    vertical_fft(&pingpong2, d_dy, d_pingpong2, stream2);
    permute(d_dxdz, stream1);
    permute(d_dy, stream2);

    // Combine
    kernel::fft::combine << < grid, block >> > (dptr, d_dy, d_dxdz);
    cuda_debug_check();

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    // Set vbo time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&vbo_time, start, stop);

    if (MODE == CORRECTNESS) {
        cudaDeviceSynchronize();
        int sizes = N * N * sizeof(float3);
        h_data = (float3*)malloc(sizes);
        checkCudaErrors(cudaMemcpy(h_data, dptr, sizes, cudaMemcpyDeviceToHost));
        return h_data;
    }

    // unmap buffer object
    if (vbo_resource != NULL)
        checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
    return SUCCESS;
}