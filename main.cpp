#include "utils.h"
#include "graphics.h"

void run_cpu(int argc, char** argv) {

    // Initialize OpenGL
    if (!graphics::cpu::initGL(&argc, argv)) {
        fprintf(stderr, "Error: OpenGL failed to initialize\n");
        exit(1);
    }

    // Register callbacks
    glutDisplayFunc(graphics::cpu::display);
    glutCloseFunc(graphics::cpu::cleanup);

    // Create initial heightmap
    ocean::cpu::create_htilde_0();

    // Precalculate Twiddle and Index Matrix
    if (VBO_MODE == VBO_FFT)
        ocean::cpu::fft::calculate_butterfly();

    // Create VBO
    vbo::cpu::createVBO(&vbo::vbo);

    // Start main loop
    glutMainLoop();
}

void run_gpu(int argc, char** argv) {
    // Initialize OpenGL
    if (!graphics::gpu::initGL(&argc, argv)) {
        fprintf(stderr, "Error: OpenGL failed to initialize\n");
        exit(1);
    }

    // register callbacks
    glutDisplayFunc(graphics::gpu::display);
    glutCloseFunc(graphics::gpu::cleanup);

    // Create initial heightmap
    ocean::gpu::create_htilde_0();

    if (VBO_MODE == VBO_FFT) {
        ocean::gpu::fft::calculate_reverse();
        ocean::gpu::fft::calculate_butterfly();
    }

    // Create VBO
    vbo::gpu::createVBO(&vbo::vbo, &vbo::cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    // Start main loop
    glutMainLoop();
}

void check_correctness() {
    // Correctness Flags
    /*
    int htilde0_correct = 1;
    int reversed_correct = 1;
    int butterfly_correct = 1;
    int hfft_correct = 1;
    int vfft_correct = 1;
    int permute_correct = 1;
    int vbo_correct = 1;

    // Check correctness of CPU and GPU htilde0
    vertex* h_vertices = ocean::cpu::create_htilde_0();
    vertex* d_vertices = ocean::gpu::create_htilde_0();

    // Compare values
    complex max_diff(0.001f, 0.001f);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            vertex v1 = h_vertices[index];
            vertex v2 = d_vertices[index];

            if ((v1.htilde0 - v1.htilde0) > max_diff || (v1.htild0_conj - v2.htild0_conj) > max_diff) {
                fprintf(stderr, "htilde0 not matching at index %d\n", index);
                htilde0_correct = 0;
            }
        }
    }

    // Final result
    if (htilde0_correct)
        printf("htilde0 is correct\n");

    
    // Check correctness of the reversed array
    int* d_reverse = ocean::gpu::fft::calculate_reverse();

    for (int x = 0; x < N; ++x) {
        if (d_reverse[x] != ocean::cpu::fft::reversed(x)) {
            fprintf(stderr, "reversed not matching at index %d\n", x);
            reversed_correct = 0;
        }
    }

    if(reversed_correct)
        printf("reversed is correct\n");

    // Check correctess of the butterfly matrix
    butterfly* h_butterfly = ocean::cpu::fft::calculate_butterfly();
    butterfly* d_butterfly = ocean::gpu::fft::calculate_butterfly();

    // Compare values
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < LOGN; ++x) {
            int index = y * LOGN + x;
            butterfly b1 = h_butterfly[index];
            butterfly b2 = d_butterfly[index];

            if ((b1.twiddle - b2.twiddle) > max_diff || b1.index1 != b2.index1 || b1.index2 != b2.index2) {
                fprintf(stderr, "butterfly matrix not matching at index %d\n", index);
                butterfly_correct = 0;
            }
        }
    }

    if (butterfly_correct)
        printf("butterfly matrix is correct\n");


    // Check correctness of all vbo calculations at a random time
    float time = (float)(rand() % 3600);
    time = 0;
    float3 pos_diff = make_float3(0.001f, 0.001f, 0.001f);


    // Check heights
    //ocean::cpu::h(vector2(0,0), time);
    //complex* direct_htilde = htildeM;
    complex* fft_htilde = ocean::cpu::fft::h(time);
    complex* d_fft_htilde = ocean::gpu::fft::h(time);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            complex c1 = d_fft_htilde[index];
            complex c2 = fft_htilde[index];

            if ((c1 - c2) > max_diff || (c1 - c2) > max_diff) {
                fprintf(stderr, "htilde not matching at index %d\n", index);
                htilde0_correct = 0;
            }
        }
    }

    // Check horizontal fft
    int pingpong = 0;
    complex* d_hfft = ocean::gpu::fft::horizontal_fft(&pingpong, d_dy);

    pingpong = 0;
    complex* h_hfft = ocean::cpu::fft::horizontal_fft(&pingpong, Dy, log2(N));

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            complex c1 = d_hfft[index];
            complex c2 = h_hfft[index];

            if ((c1 - c2) > max_diff || (c1 - c2) > max_diff) {
                fprintf(stderr, "hfft not matching at index %d\n", index);
                hfft_correct = 0;
            }
        }
    }

    if (hfft_correct)
        printf("HFFT is correct\n");


    // Check vertical fft
    pingpong = 0;
    complex* d_vfft = ocean::gpu::fft::vertical_fft(&pingpong, d_dy);

    pingpong = 0;
    complex* h_vfft = ocean::cpu::fft::vertical_fft(&pingpong, Dy, log2(N));

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            complex c1 = d_vfft[index];
            complex c2 = h_vfft[index];

            if ((c1 - c2) > max_diff || (c1 - c2) > max_diff) {
                fprintf(stderr, "htilde not matching at index %d\n", index);
                vfft_correct = 0;
            }
        }
    }

    if (vfft_correct)
        printf("VFFT is correct\n");


    // Check permute
    complex* h_p = ocean::cpu::fft::permute(Dy);
    complex* d_p = ocean::gpu::fft::permute(d_dy);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            complex c1 = h_p[index];
            complex c2 = d_p[index];

            if ((c1 - c2) > max_diff || (c1 - c2) > max_diff) {
                fprintf(stderr, "htilde not matching at index %d\n", index);
                permute_correct = 0;
            }
        }
    }

    if (permute_correct)
        printf("Permute is correct\n");


    // Get correct direct calculation
    //float3* cpu_direct = ocean::cpu::direct::calculate_vbo(NULL, time);

    float3* cpu_fft = ocean::cpu::fft::calculate_vbo(NULL, time);

    //vbo::gpu::createVBO(&vbo::vbo, &vbo::cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    float3* gpu_fft = ocean::gpu::fft::calculate_vbo(NULL, time);

    // Compare CPU results
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            float3 p1 = gpu_fft[index];
            float3 p2 = cpu_fft[index];

            if (abs(p1.x - p2.x) > pos_diff.x || abs(p1.y - p2.y) > pos_diff.y || abs(p1.z - p2.z) > pos_diff.z) {
                fprintf(stderr, "GPU and CPU ffts do not match at index %d\n", index);
                vbo_correct = 0;
            }
        }
    }

    if (vbo_correct)
        printf("Final vbo is correct\n");
*/
}

int main(int argc, char** argv)
{
    shared::generate_random_numbers();

    switch (MODE) {
    case CPU: 
    case CPU_PLUS:
        run_cpu(argc, argv); break;
    case GPU: run_gpu(argc, argv); break;
    case CORRECTNESS: check_correctness(); break;
    default: fprintf(stderr, "Error: MODE %d not valid\n", MODE); return UNSUCCESSFUL;
    }
      
    return 0;
}