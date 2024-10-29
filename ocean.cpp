#include "ocean.h"

int log_2_N = (int)log2(N);

//vertex vertices[N * N];
vertex* vertices;

// Correctness
//complex htildeM[N * N];
complex* htildeM;

// FFT
//complex Dx_Dz[N * N];
complex* Dx_Dz;
//complex Dy[N * N];
complex* Dy;
butterfly* butterflyMatrix;
//complex PingPong[N * N];
complex* PingPong;

float ocean::cpu::phillips(int n_prime, int m_prime) {
    vector2 k(PI * (2.0f * n_prime - N) / LENGTH, PI * (2.0f * m_prime - N) / LENGTH);
    float k_len = k.len();

    if (k_len < K_CLAMP)
        k_len = K_CLAMP;

    float k_len2 = k_len * k_len;
    float k_len4 = k_len2 * k_len2;

    float k_dot_w = k.unit() * W.unit();
    float k_dot_w2 = k_dot_w * k_dot_w;

    float L2 = L * L;

    return A * exp(-1.0f / (k_len2 * L2)) / k_len4 * k_dot_w2 * exp(-k_len2 * L2 * 0.00001);
}

// Create global array of initial height
vertex* ocean::cpu::create_htilde_0() {
    // Initialize butterflyMatrix
    butterflyMatrix = (butterfly*)malloc(log_2_N * N * sizeof(butterfly));

    // Create htilde0
    for (int m_prime = 0; m_prime < N; ++m_prime) {
        for (int n_prime = 0; n_prime < N; ++n_prime) {
            int index = m_prime * N + n_prime;

            float4 randoms;
            switch (MODE) {
            case CPU_PLUS:
            case CORRECTNESS: 
                randoms = make_float4(shared::h_random_numbers[index],
                    shared::h_random_numbers[index + N],
                    shared::h_random_numbers[index + 2 * N],
                    shared::h_random_numbers[index + 3 * N]);
                break;
            case CPU:
                randoms = make_float4(shared::h_random_numbers[index * 4],
                    shared::h_random_numbers[index * 4 + 1],
                    shared::h_random_numbers[index * 4 + 2],
                    shared::h_random_numbers[index * 4 + 3]);
            }

            float phillips1 = sqrt(phillips(n_prime, m_prime) / 2.0f);
            float phillips2 = sqrt(phillips(-n_prime, -m_prime) / 2.0f);

            vertices[index].htilde0 = complex(phillips1 * randoms.x, phillips1 * randoms.y);
            vertices[index].htild0_conj = complex(phillips2 * randoms.z, phillips2 * randoms.w);
        }
    }

    switch (MODE) {
    case CORRECTNESS: return vertices;
    }
    return SUCCESS;
}

// Gets value of the dispersion using the magnitude of the vector k
float ocean::cpu::dispersion(int n_prime, int m_prime) {
    vector2 k(PI * (2 * n_prime - N) / LENGTH, PI * (2 * m_prime - N) / LENGTH);
    return sqrt(g * k.len());
}

// Get partial height of a vertex at time t
complex ocean::cpu::htilde(int n_prime, int m_prime, float t) {
    int index = m_prime * N + n_prime;

    complex tilde0 = vertices[index].htilde0;
    complex tilde0_conj = vertices[index].htild0_conj;

    float omegat = dispersion(n_prime, m_prime) * t;

    complex c(cos(omegat), sin(omegat));

    complex htilde0_c0 = tilde0 * c;
    complex htilde0_conj_c1 = tilde0_conj * c.conj();
    
    return htilde0_c0 + htilde0_conj_c1;
}

// Height at a vector at point x at time t
float4 ocean::cpu::h(vector2 x, float t) {
    complex h(0.0f, 0.0f);
    complex D(0.0f, 0.0f);

    float kz, kx;

    for (int m_prime = 0; m_prime < N; ++m_prime) {
        for (int n_prime = 0; n_prime < N; ++n_prime) {
            complex tilde = htilde(n_prime, m_prime, t);

            if (MODE == CORRECTNESS) {
                htildeM[m_prime * N + n_prime] = tilde;
                continue;
            }

            vector2 k(2.0f * PI * (n_prime - N / 2.0f) / LENGTH, 2.0f * PI * (m_prime - N / 2.0f) / LENGTH);

            float k_len = k.len();
            float k_dot_x = k * x;

            complex c(cos(k_dot_x), sin(k_dot_x));

            complex tilde_c = tilde * c;
            h += tilde_c;

            if (k_len < 0.000001f)
                continue;

            complex d(k.x / k_len * tilde_c.r, k.y / k_len * tilde_c.i);

            D += d;
        }
    }

    return make_float4(h.r, h.i, D.r, D.i);
}

// Called by the current display function
// Returns a filled vbo
float3* ocean::cpu::direct::calculate_vbo(GLuint* vbo, float time) {
    auto start = std::chrono::high_resolution_clock::now();

    float3* data = (float3*)malloc(N * N * sizeof(float3));

    for (int m = 0; m < N; ++m) {
        for (int n = 0; n < N; ++n) {
            int index = m * N + n;

            vector2 x(n - (N / 2) / LENGTH, m - (N / 2) / LENGTH);
            float4 h_d = h(x, time);

            float height = h_d.x;

            float x_pos = x.x - h_d.z;
            float z_pos = x.y - h_d.w;

            data[index] = make_float3(x_pos, height, z_pos);
        }
    }

    if (vbo != NULL)
        glBufferData(GL_ARRAY_BUFFER, N * N * sizeof(float3), data, GL_DYNAMIC_DRAW);

    auto end = std::chrono::high_resolution_clock::now();
    vbo_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;

    switch (MODE) {
    case CORRECTNESS: 
        return data;
    }
    free(data);
    return SUCCESS;
}




// FFT calculate
int ocean::cpu::fft::reversed(int i) {
    unsigned int res = 0;
    for (int j = 0; j < LOGN; j++) {
        res = (res << 1) + (i & 1);
        i >>= 1;
    }

    return res;
}

butterfly* ocean::cpu::fft::calculate_butterfly() {
    // Precompute Twiddle
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < LOGN; ++x) {
            int index = y * LOGN + x;

            float k = fmodf(y * ((float)N / pow(2, x + 1)), N);
            float e_factor = 2 * PI * k / (float)N;

            float tr = cos(e_factor);
            float ti = sin(e_factor);

            if (abs(tr) < 0.00001f)
                tr = 0;
            if (abs(ti) < 0.00001f)
                ti = 0;

            complex twiddle(tr, -ti);

            int span = int(pow(2, x));
            int butterflywing = 0;

            if (fmodf(y, pow(2, x + 1)) < pow(2, x))
                butterflywing = 1;

            butterfly b;
            b.twiddle = twiddle;

            if (x == 0) {
                if (butterflywing == 1) {
                    b.index1 = reversed(y);
                    b.index2 = reversed(y + 1);
                }
                else {
                    b.index1 = reversed(y - 1);
                    b.index2 = reversed(y);
                }
            }
            else {
                if (butterflywing == 1) {
                    b.index1 = y;
                    b.index2 = y + span;
                }
                else {
                    b.index1 = y - span;
                    b.index2 = y;
                }
            }

            butterflyMatrix[index] = b;
        }
    }

    if (MODE == CORRECTNESS)
        return butterflyMatrix;

    return SUCCESS;
}

complex* ocean::cpu::fft::h(float time) {
    for (int m_prime = 0; m_prime < N; ++m_prime) {
        for (int n_prime = 0; n_prime < N; ++n_prime) {
            int index = m_prime * N + n_prime;
            vector2 k(2.0 * PI * (n_prime - (float)N / 2) / LENGTH, 2.0 * PI * (m_prime - (float)N / 2) / L);

            float phase = dispersion(n_prime, m_prime) * time;

            complex exponent = complex(cos(phase), sin(phase));
            complex h = vertices[index].htilde0 * exponent + vertices[index].htild0_conj * exponent.conj();
            Dy[index] = h;

            float len = k.len();
            if (len < K_CLAMP) {
                Dx_Dz[index] = complex(0.0f, 0.0f);
                continue;
            }

            complex ih = complex(-h.i, h.r);

            complex displacementX = ih * (k.x / len);
            complex displacementZ = ih * (k.y / len);

            Dx_Dz[index] = complex(displacementX.r - displacementZ.i, displacementX.i + displacementZ.r);
        }
    }

    if (MODE == CORRECTNESS)
        return Dy;

    return SUCCESS;
}

complex* ocean::cpu::fft::horizontal_fft(int* pingpong, complex* input, int logSize) {
    for (int i = 0; i < logSize; i++) {
        *pingpong ^= 1;

        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int index = y * N + x;
                butterfly b = butterflyMatrix[x * LOGN + i];

                if (*pingpong) {
                    complex b01 = input[y * N + b.index1];
                    complex b02 = input[y * N + b.index2];

                    PingPong[index] = b01 + b.twiddle * b02;
                }
                else {
                    complex b11 = PingPong[y * N + b.index1];
                    complex b12 = PingPong[y * N + b.index2];

                    input[index] = b11 + b.twiddle * b12;
                }
            }
        }
    }

    if (MODE == CORRECTNESS)
        return input;

    return SUCCESS;
}

complex* ocean::cpu::fft::vertical_fft(int* pingpong, complex* input, int logSize) {
    for (int i = 0; i < logSize; i++) {
        *pingpong ^= 1;
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int index = y * N + x;
                butterfly b = butterflyMatrix[y * LOGN + i];

                if (*pingpong) {
                    complex b01 = input[b.index1 * N + x];
                    complex b02 = input[b.index2 * N + x];

                    PingPong[index] = b01 + b.twiddle * b02;
                }
                else {
                    complex b01 = PingPong[b.index1 * N + x];
                    complex b02 = PingPong[b.index2 * N + x];

                    input[index] = b01 + b.twiddle * b02;
                }
            }
        }
    }

    if (MODE == CORRECTNESS)
        return input;

    return SUCCESS;
}

complex* ocean::cpu::fft::permute(complex* result) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            result[y * N + x] = result[y * N + x] * (1.0 - 2.0 * ((x + y) % 2)) / (float)(N);
        }
    }

    if (MODE == CORRECTNESS)
        return result;

    return SUCCESS;
}

float3* ocean::cpu::fft::calculate_vbo(GLuint* vbo, float t) {
    auto start = std::chrono::high_resolution_clock::now();
    float3* data = (float3*)malloc(N * N * sizeof(float3));

    // Create height map
     h(t);

    int logSize = (int)log2(N);
    int pingpong = 0;

    // Horizontal Displacement
    horizontal_fft(&pingpong, Dx_Dz, logSize);
    vertical_fft(&pingpong, Dx_Dz, logSize);

    // Permute
    permute(Dx_Dz);

    // Vertical Displacement
    horizontal_fft(&pingpong, Dy, logSize);
    vertical_fft(&pingpong, Dy, logSize);

    // Permute
    permute(Dy);

    // Combine
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            vector2 x_vec(x - N / 2 / LENGTH, y - N / 2 / LENGTH);

            data[index] = make_float3(x_vec.x - Dx_Dz[index].r, Dy[index].r, x_vec.y - Dx_Dz[index].r);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    vbo_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;

    if (vbo != NULL)
        glBufferData(GL_ARRAY_BUFFER, N * N * sizeof(float3), data, GL_DYNAMIC_DRAW);

    if (MODE == CORRECTNESS)
        return data;

    free(data);
    return SUCCESS;
}