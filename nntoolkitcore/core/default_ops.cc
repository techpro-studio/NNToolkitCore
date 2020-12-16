//
// Created by Alex on 07.11.2020.
//


#include "ops.h"
#include "math.h"
#include "third_party/eigen3/Eigen/Dense"

#if (__arm__) || (__arm64__)
    #include <arm_neon.h>
    #define NEON __ARM_NEON__
#else
    #define NEON 0
#endif


typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> const_map_t;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map_t;



#if NEON

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1


static float32x4_t exp_neon(float32x4_t x) {

  float32x4_t tmp, fx;

  float32x4_t one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
  float32x4_t y = vld1q_dup_f32(cephes_exp_p+0);
  float32x4_t c1 = vld1q_dup_f32(cephes_exp_p+1);
  float32x4_t c2 = vld1q_dup_f32(cephes_exp_p+2);
  float32x4_t c3 = vld1q_dup_f32(cephes_exp_p+3);
  float32x4_t c4 = vld1q_dup_f32(cephes_exp_p+4);
  float32x4_t c5 = vld1q_dup_f32(cephes_exp_p+5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x,x);
  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);

  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 - 1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 - 1.2420140846E-1
#define c_cephes_log_p4 + 1.4249322787E-1
#define c_cephes_log_p5 - 1.6668057665E-1
#define c_cephes_log_p6 + 2.0000714765E-1
#define c_cephes_log_p7 - 2.4999993993E-1
#define c_cephes_log_p8 + 3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

float32x4_t log_neon(float32x4_t x) {
  float32x4_t one = vdupq_n_f32(1);

  x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
  uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

  int32x4_t ux = vreinterpretq_s32_f32(x);

  int32x4_t emm0 = vshrq_n_s32(ux, 23);

  /* keep only the fractional part */
  ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
  ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
  x = vreinterpretq_f32_s32(ux);

  emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
  float32x4_t e = vcvtq_f32_s32(emm0);

  e = vaddq_f32(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
  float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
  x = vsubq_f32(x, one);
  e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
  x = vaddq_f32(x, tmp);

  float32x4_t z = vmulq_f32(x,x);

  float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
  y = vmulq_f32(y, x);

  y = vmulq_f32(y, z);


  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
  y = vaddq_f32(y, tmp);


  tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
  y = vsubq_f32(y, tmp);

  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
  x = vaddq_f32(x, y);
  x = vaddq_f32(x, tmp);
  x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
  return x;
}

float32x4_t neon_div(float32x4_t a, float32x4_t b){
#if __arm64__
    return vdivq_f32(a, b);
#else
    return vmulq_f32(a, vrecpeq_f32(b));
#endif
}

#define ln10 2.30258509299

float32x4_t log10_neon(float32x4_t a){
    return neon_div(log_neon(a), vdupq_n_f32(ln10));
}

void mat_mul_neon_slow(const float *a, const float *b, float *c, int M, int N, int K){
    for (int m = 0; m < M; ++m){
        for (int n = 0; n < N; ++n){
            int parts = K / 4, remaining = K % 4;
            float32x4_t sum_4 = vdupq_n_f32(0);
            for (int k = 0; k < parts; ++k){
                float32x4_t a_4 = vld1q_f32(a + (m * K + 4 * k));
                float b_arr_4 [4] = { b[4 * k * N + n], b[(4 * k + 1) * N + n],
                    b[(4 * k + 2) * N + n], b[(4 * k + 3) * N + n] };
                sum_4 = vmlaq_f32(sum_4, a_4, vld1q_f32(b_arr_4));
            }
            float sum[4];
            vst1q_f32(sum, sum_4);
            float remaining_sum = 0.0f;
            for (int r = 0; r < remaining; ++r){
                remaining_sum += a[m * K + 4 * parts + r] * b[((r + 4 * parts) * N) + n];
            }
            c[m * N + n] = sum[0] + sum[1] + sum[2] + sum[3] + remaining_sum;
        }
    }
}

#endif


static inline float op_vec_dot_c(const float *a, const float *b, int size){
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}


float op_vec_dot(const float *a, const float *b, int size) {
#if NEON
    int n = 4;
    int parts = size / n, remaining = size % n;
    float32x4_t sum_4 = vdupq_n_f32(0);

    for (int i = 0; i < parts; ++i){
        sum_4 = vmlaq_f32(sum_4, vld1q_f32(a + n * i), vld1q_f32(b + n * i));
    }

    float sum_4_arm[4];
    vst1q_f32(sum_4_arm, sum_4);
    return op_vec_dot_c(a + 4 * parts, b + 4 * parts, remaining) + sum_4_arm[0] + sum_4_arm[1] + sum_4_arm[2] + sum_4_arm[3];
#else
    return op_vec_dot_c(a, b, size);
#endif
}

static inline void op_vec_add_c(const float *a, const float * b, float *c, int size) {
    for (int i = 0; i < size; ++i){
        c[i] = a[i] + b[i];
    }
}



void op_vec_add(const float *a, const float * b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vaddq_f32(vld1q_f32(a + 4 * i), vld1q_f32(b + 4 * i)));
    }
    op_vec_add_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_add_c(a, b, c, size);
#endif
}


static inline void op_vec_sqrt_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = sqrtf(a[i]);
    }
}

void op_vec_sqrt(const float *a, float *c, int size){
#if NEON & __arm64
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vsqrtq_f32(vld1q_f32(a + 4 * i)));
    }
    op_vec_sqrt_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_sqrt_c(a, c, size);
#endif
}

static inline void op_vec_exp_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = expf(a[i]);
    }
}

void op_vec_exp(const float *a, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, exp_neon(vld1q_f32(a + 4 * i)));
    }
    op_vec_exp_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_exp_c(a, c, size);
#endif
}

static inline void op_vec_pow_c(const float *a, const float *b, float *c, int size) {
    for (int i = 0; i < size; ++i){
        c[i] = powf(a[i], b[i]);
    }
}
// x ^^ m = exp(m * log(x))

void op_vec_pow(const float *a, const float *b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, exp_neon(vmulq_f32(vld1q_f32(b + 4 * i), log_neon(vld1q_f32(a + 4 * i)))));
    }
    op_vec_pow_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_pow_c(a, b, c, size);
#endif
}

static inline void op_vec_pow_sc_c(const float *a, const float b, float *c, int size) {
    for (int i = 0; i < size; ++i){
        c[i] = powf(a[i], b);
    }
}

void op_vec_pow_sc(const float *a, const float b, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, exp_neon(vmulq_f32(vdupq_n_f32(b), log_neon(vld1q_f32(a + 4 * i)))));
    }
    op_vec_pow_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_pow_sc_c(a, b, c, size);
#endif
}


static inline void op_vec_log_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = logf(a[i]);
    }
}

void op_vec_log(const float *a, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, log_neon(vld1q_f32(a + 4 * i)));
    }
    op_vec_log_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_log_c(a, c, size);
#endif
}


static inline void op_vec_reciprocal_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = 1 / a[i];
    }
}

void op_vec_reciprocal(const float *a, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        float32x4_t inp = vld1q_f32(a + 4 * i);
        float32x4_t reciprocal = vrecpeq_f32(inp);
        float32x4_t result = vmulq_f32(vrecpsq_f32(inp, reciprocal), reciprocal);
        vst1q_f32(c + i * 4, result);
    }
    op_vec_reciprocal_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_reciprocal_c(a, c, size);
#endif
}

static inline void op_vec_max_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = fmaxf(a[i], b[i]);
    }
}

void op_vec_max(const float *a, const float *b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vmaxq_f32(vld1q_f32(a + 4 * i), vld1q_f32(b + 4 * i)));
    }
    op_vec_max_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_max_c(a, b, c, size);
#endif
}

static inline void op_vec_min_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = fminf(a[i], b[i]);
    }
}

void op_vec_min(const float *a, const float *b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vminq_f32(vld1q_f32(a + 4 * i), vld1q_f32(b + 4 * i)));
    }
    op_vec_min_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_min_c(a, b, c, size);
#endif
}

static inline void op_vec_tanh_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = tanhf(a[i]);
    }
}


void op_vec_tanh(const float *a, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t minus_one = vdupq_n_f32(-1.0f);
    for (int i = 0; i < parts; ++i){
        float32x4_t x = vld1q_f32(a + 4 * i);
        float32x4_t e_x = exp_neon(x);
        float32x4_t e_minus_x = exp_neon(vmulq_f32(x, minus_one));
        vst1q_f32(c + i * 4, neon_div(vsubq_f32(e_x, e_minus_x), vaddq_f32(e_x, e_minus_x)));
    }
    op_vec_tanh_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_tanh_c(a, c, size);
#endif
}


static inline void op_vec_clamp_c(const float *a, float *c, float min, float max, int size){
    for (int i = 0; i < size; ++i){
        c[i] = fmaxf(fminf(a[i], max), min);
    }
}

void op_vec_clamp(const float *a, float *c, float min, float max, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t min_4 = vdupq_n_f32(min);
    float32x4_t max_4 = vdupq_n_f32(max);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vmaxq_f32(vminq_f32(vld1q_f32(a + 4 * i), max_4), min_4));
    }
    op_vec_clamp_c(a + parts * 4, c + parts * 4, min, max, remaining);
#else
    op_vec_clamp_c(a, c, min, max, size);
#endif
}

static void op_vec_max_sc_c(const float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = fmaxf(a[i], b);
    }
}

void op_vec_max_sc(const float *a, float b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t max_4 = vdupq_n_f32(b);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vmaxq_f32(vld1q_f32(a + 4 * i), max_4));
    }
    op_vec_max_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_max_sc_c(a, b, c, size);
#endif
}

static void op_vec_add_sc_c(const float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] + b;
    }
}

void op_vec_add_sc(const float *a, float b, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t b_4 = vdupq_n_f32(b);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vaddq_f32(vld1q_f32(a + 4 * i), b_4));
    }
    op_vec_add_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_add_sc_c(a, b, c, size);
#endif
}

static void op_vec_sub_sc_c(const float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] - b;
    }
}

void op_vec_sub_sc(const float *a, float b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t b_4 = vdupq_n_f32(b);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vsubq_f32(vld1q_f32(a + 4 * i), b_4));
    }
    op_vec_sub_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_sub_sc_c(a, b, c, size);
#endif
}

static void op_vec_mul_sc_c(const float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] * b;
    }
}

void op_vec_mul_sc(const float *a, float b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t b_4 = vdupq_n_f32(b);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vmulq_f32(vld1q_f32(a + 4 * i), b_4));
    }
    op_vec_mul_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_mul_sc_c(a, b, c, size);
#endif
}



static void op_vec_div_sc_c(const float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] / b;
    }
}

void op_vec_div_sc(const float *a, float b, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t b_4 = vdupq_n_f32(b);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, neon_div(vld1q_f32(a + 4 * i), b_4));
    }
    op_vec_div_sc_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_div_sc_c(a, b, c, size);
#endif
}

static void op_vec_neg_c(const float *a, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = -a[i];
    }
}

void op_vec_neg(const float *a, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vnegq_f32(vld1q_f32(a + 4 * i)));
    }
    op_vec_neg_c(a + parts * 4, c + parts * 4, remaining);
#else
    op_vec_neg_c(a, c, size);
#endif
}



static void op_vec_div_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] / b[i];
    }
}

void op_vec_div(const float *a, const float *b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, neon_div(vld1q_f32((a + 4 * i)), vld1q_f32(b + 4 * i)));
    }
    op_vec_div_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_div_c(a, b, c, size);
#endif
}

static void op_vec_mul_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] * b[i];
    }
}

void op_vec_mul(const float *a, const float *b, float* c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vmulq_f32(vld1q_f32(a + 4 * i), vld1q_f32(b + 4 * i)));
    }
    op_vec_mul_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_mul_c(a, b, c, size);
#endif
}

static void op_vec_sub_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] - b[i];
    }
}

void op_vec_sub(const float *a, const float *b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + i * 4, vsubq_f32(vld1q_f32(a + 4 * i), vld1q_f32(b + 4 * i)));
    }
    op_vec_sub_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_sub_c(a, b, c, size);
#endif
}

static void op_vec_sum_c(const float *a, float *c, int size){
    float sum = 0.0f;
    for (int i = 0; i < size; ++i){
        sum += a[i];
    }
    *c = sum;
}


void op_vec_sum(const float *a, float* c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t sum = vdupq_n_f32(0);
    for (int i = 0; i < parts; ++i){
        sum = vaddq_f32(sum, vld1q_f32(a + 4 * i));
    }
    float sum4[4];
    vst1q_f32(sum4, sum);
    float c1 = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    float c2 = 0.0f;
    op_vec_sum_c(a + parts * 4, &c2, remaining);
    *c = c1 + c2;
#else
    op_vec_sum_c(a, c, size);
#endif
}


static void op_vec_magn_sq_c(const float *a, const float *b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] * a[i] + b[i] * b[i];
    }
}

void op_vec_magn_sq(float *a, float *b, float *c, int size) {
#if NEON
    int parts = size / 4, remaining = size % 4;
    for (int i = 0; i < parts; ++i){
        float32x4_t a_4 = vld1q_f32(a + 4 * i);
        float32x4_t b_4 = vld1q_f32(b + 4 * i);
        vst1q_f32(c + i * 4, vaddq_f32(vmulq_f32(a_4, a_4), vmulq_f32(b_4, b_4)));
    }
    op_vec_magn_sq_c(a + parts * 4, b + parts * 4, c + parts * 4, remaining);
#else
    op_vec_magn_sq_c(a, b, c, size);
#endif
}

void op_vec_db_c(float *a, float b, float *c, int size){
    for (int i = 0; i < size; ++i){
        c[i] = 10 * log10f((a[i] / b));
    }
}

void op_vec_db(float *a, float b, float *c, int size){
#if NEON
    int parts = size / 4, remaining = size % 4;
    float32x4_t b_4 = vdupq_n_f32(b);
    float32x4_t ten = vdupq_n_f32(10.0f);
    for (int i = 0; i < parts; ++i){
        vst1q_f32(c + 4 * i, vmulq_f32(ten, log10_neon(neon_div(vld1q_f32(a + 4 * i), b_4))));
    }
    op_vec_db_c(a + parts * 4, b, c + parts * 4, remaining);
#else
    op_vec_db_c(a, b, c, size);
#endif
}


void op_mat_mul_c(const float *a, const float *b, float *c, int M, int N, int K) {
    for (int m = 0; m < M; ++m){
        for (int n = 0; n < N; ++n){
            float c_mn = 0.0f;
            for (int k = 0; k < K; ++k){
                c_mn += a[m * K + k] * b [k * N + n];
            }
            c[m * N + n] = c_mn;
        }
    }
}

void op_mat_transp_c(const float *a, float *b, int m, int n) {
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            b[i * n + j] = a[j * m + i];
        }
    }
}



void op_mat_mul(const float *a, const float *b, float *c, int M, int N, int K) {
    const_map_t mA(a, M, K);
    const_map_t mB(b, K, N);
    map_t mC(c, M, N);
    if (N == 1) {
        mC.col(0).noalias() = mA * mB.col(0);
    } else if (M == 1) {
        mC.row(0).noalias() = mA.row(0) * mB;
    } else {
        mC.noalias() = mA * mB;
    }
}


void op_mat_transp(const float *a, float *b, int M, int N) {
    const_map_t mA(a, N, M);
    map_t mB(b, M, N);
    mB.noalias() = mA.transpose();
}










