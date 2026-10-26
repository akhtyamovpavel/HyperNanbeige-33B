#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ggml_context;
struct ggml_tensor;

typedef ggml_tensor * (*fn_ggml_mul_mat_t)(ggml_context *, ggml_tensor *, ggml_tensor *);
typedef ggml_tensor * (*fn_ggml_view_2d_t)(ggml_context *, ggml_tensor *, int64_t, int64_t, size_t, size_t);
typedef ggml_tensor * (*fn_ggml_scale_bias_t)(ggml_context *, ggml_tensor *, float, float);
typedef ggml_tensor * (*fn_ggml_pad_t)(ggml_context *, ggml_tensor *, int, int, int, int);
typedef ggml_tensor * (*fn_ggml_concat_t)(ggml_context *, ggml_tensor *, ggml_tensor *, int);

static fn_ggml_mul_mat_t    real_ggml_mul_mat    = nullptr;
static fn_ggml_view_2d_t    real_ggml_view_2d    = nullptr;
static fn_ggml_scale_bias_t real_ggml_scale_bias = nullptr;
static fn_ggml_pad_t        real_ggml_pad        = nullptr;
static fn_ggml_concat_t     real_ggml_concat     = nullptr;

static ggml_tensor * g_target_output = nullptr;
static ggml_tensor * g_draft_output  = nullptr;
static int64_t       g_k_head        = 32768;
static bool          g_share_target  = false;
static bool          g_logged        = false;

static void init_syms() {
    if (real_ggml_mul_mat) return;
    real_ggml_mul_mat    = (fn_ggml_mul_mat_t)   dlsym(RTLD_NEXT, "ggml_mul_mat");
    real_ggml_view_2d    = (fn_ggml_view_2d_t)   dlsym(RTLD_NEXT, "ggml_view_2d");
    real_ggml_scale_bias = (fn_ggml_scale_bias_t)dlsym(RTLD_NEXT, "ggml_scale_bias");
    real_ggml_pad        = (fn_ggml_pad_t)       dlsym(RTLD_NEXT, "ggml_pad");
    real_ggml_concat     = (fn_ggml_concat_t)    dlsym(RTLD_NEXT, "ggml_concat");

    if (const char * env_k = getenv("FRSPEC_K_HEAD")) {
        g_k_head = atoll(env_k);
    }
    if (const char * env_s = getenv("FRSPEC_SHARE_TARGET")) {
        g_share_target = (atoi(env_s) != 0);
    }
}

extern "C" ggml_tensor * ggml_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    init_syms();
    if (!a || !b) {
        return real_ggml_mul_mat(ctx, a, b);
    }

    const char * name = (const char *)a + 0x100;
    int32_t b_op = *(const int32_t *)((const char *)b + 0x50);
    if (b_op != 0 && strcmp(name, "output.weight") == 0) {
        int64_t ne0 = *(const int64_t *)((const char *)a + 0x10);
        int64_t ne1 = *(const int64_t *)((const char *)a + 0x18);
        if (ne0 == 7168 && ne1 == 166912) {
            if (g_target_output == nullptr) {
                g_target_output = a;
                fprintf(stderr, "[FR-Spec] Registered Target 33B output.weight=%p\n", (void *)a);
                return real_ggml_mul_mat(ctx, a, b);
            }
            if (a != g_target_output) {
                if (g_draft_output == nullptr) {
                    g_draft_output = a;
                    fprintf(
                        stderr,
                        "[FR-Spec] Intercepted Draft output.weight=%p -> slicing to Top-%lld + 896 tail (share_target=%d)\n",
                        (void *)a,
                        (long long)g_k_head,
                        (int)g_share_target
                    );
                }
                ggml_tensor * w_src = (g_share_target && g_target_output) ? g_target_output : a;
                size_t nb1 = *(const size_t *)((const char *)w_src + 0x38);

                const int64_t tail_start = 166016;
                const int64_t tail_len   = 166912 - tail_start; // 896
                const int     pad_len    = (int)(tail_start - g_k_head);

                ggml_tensor * w_head = real_ggml_view_2d(ctx, w_src, ne0, g_k_head, nb1, 0);
                ggml_tensor * w_tail = real_ggml_view_2d(ctx, w_src, ne0, tail_len, nb1, (size_t)tail_start * nb1);

                ggml_tensor * l_head = real_ggml_mul_mat(ctx, w_head, b);
                ggml_tensor * l_tail = real_ggml_mul_mat(ctx, w_tail, b);

                ggml_tensor * h_bias = real_ggml_scale_bias(ctx, l_head, 1.0f, 100.0f);
                ggml_tensor * t_bias = real_ggml_scale_bias(ctx, l_tail, 1.0f, 100.0f);

                ggml_tensor * h_pad  = real_ggml_pad(ctx, h_bias, pad_len, 0, 0, 0);
                ggml_tensor * res    = real_ggml_concat(ctx, h_pad, t_bias, 0);
                if (!g_logged) {
                    g_logged = true;
                    fprintf(stderr, "[FR-Spec] Active graph built: [0..%lld] + [166016..166912] (pad=%d)\n",
                            (long long)g_k_head, pad_len);
                }
                return res;
            }
        }
    }
    return real_ggml_mul_mat(ctx, a, b);
}
