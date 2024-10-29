#include "ocean_kernel.h"

float kernel::phillips(int n_prime, int m_prime) {
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

__global__ void kernel::create_htilde_0_kernel(vertex *vertices, float* rand_nums) {
    int m_prime = blockIdx.y * blockDim.y + threadIdx.y;
    int n_prime = blockIdx.x * blockDim.x + threadIdx.x;

    int index = m_prime * N + n_prime;

    float4 randoms = make_float4(rand_nums[index],
        rand_nums[index + N],
        rand_nums[index + 2 * N],
        rand_nums[index + 3 * N]);

    float phillips1 = sqrt(phillips(n_prime, m_prime) / 2.0f);
    float phillips2 = sqrt(phillips(-n_prime, -m_prime) / 2.0f);

    vertices[index].htilde0 = complex(phillips1 * randoms.x, phillips1 * randoms.y);
    vertices[index].htild0_conj = complex(phillips2 * randoms.z, phillips2 * randoms.w);
}

__device__ float kernel::dispersion(int n_prime, int m_prime) {
    vector2 k(PI * (2 * n_prime - N) / LENGTH, PI * (2 * m_prime - N) / LENGTH);
    return sqrt(g * k.len());
}

__device__ complex kernel::htilde(int n_prime, int m_prime, float t, complex tilde0, complex tilde0_conj) {
    float omegat = dispersion(n_prime, m_prime) * t;

    complex c(cos(omegat), sin(omegat));
    complex htilde0_c0 = tilde0 * c;
    complex htilde0_conj_c1 = tilde0_conj * c.conj();

    return htilde0_c0 + htilde0_conj_c1;
}

// Direct One Step Functions
__device__ float3 kernel::direct_one_step::h_D(vector2 x, float t, vertex* vertices) {
    complex h(0.0f, 0.0f);
    complex D(0.0f, 0.0f);

    for (int m_prime = 0; m_prime < N; ++m_prime) {
        for (int n_prime = 0; n_prime < N; ++n_prime) {
            vector2 k(2.0f * PI * (n_prime - N / 2.0f) / LENGTH, 2.0f * PI * (m_prime - N / 2.0f) / LENGTH);

            float k_len = k.len();
            float k_dot_x = k * x;

            complex c(cos(k_dot_x), sin(k_dot_x));

            vertex cur = vertices[m_prime * N + n_prime];
            complex tilde = htilde(n_prime, m_prime, t, cur.htilde0, cur.htild0_conj);

            complex tilde_c = tilde * c;

            h += tilde_c;

            if (k_len < 0.000001f)
                continue;

            complex d(k.x / k_len * tilde_c.r, k.y / k_len * tilde_c.i);
            D += d;
        }
    }

    return make_float3(h.r, D.r, D.i);
}

__global__ void kernel::direct_one_step::calculate_vbo(float3* data, float time, vertex *vertices) {
    // Get thread id
    int index = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (threadIdx.y + blockIdx.y * blockDim.y));
    
    // Get position of current vertex
    vector2 x(blockIdx.x * blockDim.x + threadIdx.x - (N / 2) / LENGTH, blockIdx.y * blockDim.y + threadIdx.y - (N / 2) / LENGTH);
    
    // Call height function
    float3 h_d = h_D(x, time, vertices);

    // Calculate final position
    float height = h_d.x;
    float x_pos = x.x - h_d.y;
    float z_pos = x.y - h_d.z;

    // Write to global mem
    data[index] = make_float3(x_pos, height, z_pos);
}


// Direct Two Step Functions
__global__ void kernel::direct_two_step::h_D(complex* result, float t, vertex* vertices) {
    int index = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (threadIdx.y + blockIdx.y * blockDim.y));

    int m_prime = blockIdx.y * blockDim.y + threadIdx.y;
    int n_prime = blockIdx.x * blockDim.x + threadIdx.x;

    vertex cur = vertices[m_prime * N + n_prime];
    result[index] = htilde(n_prime, m_prime, t, cur.htilde0, cur.htild0_conj);
}

__global__ void kernel::direct_two_step::calculate_vbo(float3* data, float time, complex* htilde) {
    // Get thread id
    int index = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (threadIdx.y + blockIdx.y * blockDim.y));

    // Get position of current vertex
    vector2 x(blockIdx.x * blockDim.x + threadIdx.x - (N / 2) / LENGTH, blockIdx.y * blockDim.y + threadIdx.y - (N / 2) / LENGTH);

    // Call height function
    complex h(0.0f, 0.0f);
    complex D(0.0f, 0.0f);

    for (int m_prime = 0; m_prime < N; ++m_prime) {
        float kz = 2.0f * PI * (m_prime - N / 2.0f) / LENGTH;

        for (int n_prime = 0; n_prime < N; ++n_prime) {
            vector2 k(2.0f * PI * (n_prime - N / 2.0f) / LENGTH, kz);

            float k_len = k.len();
            float k_dot_x = k * x;

            complex c(cos(k_dot_x), sin(k_dot_x));

            complex tilde = htilde[m_prime * N + n_prime];
            complex tilde_c = tilde * c;

            h += tilde_c;

            if (k_len < 0.000001f)
                continue;

            complex d(k.x / k_len * tilde_c.r, k.y / k_len * tilde_c.i);
            D += d;
        }
    }

    // Calculate final position
    float height = h.r;
    float x_pos = x.x - D.r;
    float z_pos = x.y - D.i;

    // Write to global mem
    data[index] = make_float3(x_pos, height, z_pos);
}


// FFT Functions
__global__ void kernel::fft::reversed(int* result, int LOGN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int res = 0;
    for (int j = 0; j < LOGN; j++) {
        res = (res << 1) + (i & 1);
        i >>= 1;
    }
    result[blockIdx.x * blockDim.x + threadIdx.x] = res;
}

__global__ void kernel::fft::butterfly_array(butterfly* result, int* bit_reversed, int LOGN) {  
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

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
            b.index1 = bit_reversed[y];
            b.index2 = bit_reversed[y + 1];
        }
        else {
            b.index1 = bit_reversed[y - 1];
            b.index2 = bit_reversed[y];
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

    result[index] = b;
}

__global__ void kernel::fft::h(vertex* vertices, float time, complex* dy, complex* dxdz) {
    int m_prime = blockIdx.y * blockDim.y + threadIdx.y;
    int n_prime = blockIdx.x * blockDim.x + threadIdx.x;
    int index = m_prime * N + n_prime;

    float phase = dispersion(n_prime, m_prime) * time;

    complex exponent = complex(cos(phase), sin(phase));
    complex h = vertices[index].htilde0 * exponent + vertices[index].htild0_conj * exponent.conj();
    dy[index] = h;

    vector2 k(2.0 * PI * (n_prime - (float)N / 2) / LENGTH, 2.0 * PI * (m_prime - (float)N / 2) / L);
    float len = k.len();
    if (len < K_CLAMP) {
        dxdz[index] = complex(0.0f, 0.0f);
        return;
    }

    complex ih = complex(-h.i, h.r);
    complex displacementX = ih * (k.x / len);
    complex displacementZ = ih * (k.y / len);
    dxdz[index] = complex(displacementX.r - displacementZ.i, displacementX.i + displacementZ.r);

}

__global__ void kernel::fft::horizontal_fft(int stage, int pingpong, complex* pingpong0, complex* pingpong1, butterfly* precomuted, int LOGN) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * N + x;
    butterfly b = precomuted[x * LOGN + stage];

    if (pingpong) {
        complex b01 = pingpong0[y * N + b.index1];
        complex b02 = pingpong0[y * N + b.index2];

        pingpong1[index] = b01 + b.twiddle * b02;
    }
    else {
        complex b11 = pingpong1[y * N + b.index1];
        complex b12 = pingpong1[y * N + b.index2];

        pingpong0[index] = b11 + b.twiddle * b12;
    }
}

__global__ void kernel::fft::vertical_fft(int stage, int pingpong, complex* pingpong0, complex* pingpong1, butterfly* precomuted, int LOGN) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * N + x;
    butterfly b = precomuted[y * LOGN + stage];

    if (pingpong) {
        complex b01 = pingpong0[b.index1 * N + x];
        complex b02 = pingpong0[b.index2 * N + x];

        pingpong1[index] = b01 + b.twiddle * b02;
    }
    else {
        complex b01 = pingpong1[b.index1 * N + x];
        complex b02 = pingpong1[b.index2 * N + x];

        pingpong0[index] = b01 + b.twiddle * b02;
    }
}

__global__ void kernel::fft::permute(complex* result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    result[y * N + x] = result[y * N + x] * (1.0 - 2.0 * ((x + y) % 2)) / (float)(N);
}

__global__ void kernel::fft::combine(float3* data, complex* dy, complex* dxdz) {
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * N + x;
    vector2 x_vec(x - N / 2.0f / LENGTH, y - N / 2.0f / LENGTH);

    data[index] = make_float3(x_vec.x - dxdz[index].r, dy[index].r, x_vec.y - dxdz[index].r);
}