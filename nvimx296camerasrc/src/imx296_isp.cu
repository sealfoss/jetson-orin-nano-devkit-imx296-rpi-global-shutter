/* SPDX-License-Identifier: GPL-2.0-only
 * Fused IMX296 ISP kernel: RG10 (RGGB, 10-bit in 16-bit VI containers)
 *   -> linear LUT (black level + rescale)   [1024 x u16, __constant__]
 *   -> bilinear demosaic (linear domain)
 *   -> combined WB*CCM*digital-gain 3x3     [__constant__ floats]
 *   -> tone LUT (preset/contrast/brightness/knee baked) [65536 x u8, global]
 *   -> BT.601 limited-range NV12 (saturation folded into chroma coeffs)
 *   -> optional hash dither at the 8-bit quantize
 * One thread = one 2x2 Bayer quad = 4 Y samples + 1 UV pair.
 * Color math is the byte-exact CUDA port of the validated Python pipeline
 * (scripts/imx296_isp_pipeline.py); golden-tested against it.
 */
#include "imx296_isp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUCHK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return -1; } } while (0)

/* ---------------- device constants ---------------- */
__constant__ uint16_t c_lin_lut[1024];   /* p10 -> 16-bit linear            */
__constant__ float    c_mat[9];          /* combined WB*CCM*dgain, RGB      */
__constant__ float    c_yuv[12];         /* RGB8->YUV coeffs (sat folded)   */

typedef struct { float sum[3]; float cnt; } AwbStats;

/* ---------------- helpers ---------------- */
__device__ __forceinline__ int clampi(int v, int lo, int hi)
{ return v < lo ? lo : (v > hi ? hi : v); }

__device__ __forceinline__ float lin_at(const uint16_t *raw, size_t pitch16,
                                        int x, int y, int W, int H)
{
    x = clampi(x, 0, W - 1); y = clampi(y, 0, H - 1);   /* mirror-ish border */
    return (float)c_lin_lut[(raw[(size_t)y * pitch16 + x] >> 6) & 0x3ff];
}

/* Bilinear demosaic for RGGB at (x,y); returns linear RGB in 0..65535. */
__device__ void demosaic_rggb(const uint16_t *raw, size_t pitch16,
                              int x, int y, int W, int H, float rgb[3])
{
    const bool ex = !(x & 1), ey = !(y & 1);  /* even col/row */
    float c  = lin_at(raw, pitch16, x, y, W, H);
    float h  = 0.5f * (lin_at(raw, pitch16, x - 1, y, W, H) + lin_at(raw, pitch16, x + 1, y, W, H));
    float v  = 0.5f * (lin_at(raw, pitch16, x, y - 1, W, H) + lin_at(raw, pitch16, x, y + 1, W, H));
    float d  = 0.25f * (lin_at(raw, pitch16, x - 1, y - 1, W, H) + lin_at(raw, pitch16, x + 1, y - 1, W, H)
                      + lin_at(raw, pitch16, x - 1, y + 1, W, H) + lin_at(raw, pitch16, x + 1, y + 1, W, H));
    float pl = 0.5f * (h + v);                 /* plus-neighbour average    */

    if (ey && ex)        { rgb[0] = c;  rgb[1] = pl; rgb[2] = d;  } /* R site  */
    else if (ey && !ex)  { rgb[0] = h;  rgb[1] = c;  rgb[2] = v;  } /* G on R row */
    else if (!ey && ex)  { rgb[0] = v;  rgb[1] = c;  rgb[2] = h;  } /* G on B row */
    else                 { rgb[0] = d;  rgb[1] = pl; rgb[2] = c;  } /* B site  */
}

__device__ __forceinline__ float3 apply_mat(const float rgb[3])
{
    float3 o;
    o.x = fminf(fmaxf(c_mat[0]*rgb[0] + c_mat[1]*rgb[1] + c_mat[2]*rgb[2], 0.f), 65535.f);
    o.y = fminf(fmaxf(c_mat[3]*rgb[0] + c_mat[4]*rgb[1] + c_mat[5]*rgb[2], 0.f), 65535.f);
    o.z = fminf(fmaxf(c_mat[6]*rgb[0] + c_mat[7]*rgb[1] + c_mat[8]*rgb[2], 0.f), 65535.f);
    return o;
}

/* 32-bit hash -> [-0.5, 0.5) dither offset, decorrelated per pixel+frame */
__device__ __forceinline__ float dither_of(unsigned x, unsigned y, unsigned f)
{
    unsigned h = x * 0x9E3779B1u ^ y * 0x85EBCA77u ^ f * 0xC2B2AE3Du;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return (float)(h & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f;
}

/* ---------------- main fused kernel ---------------- */
template <bool OUT_RGB>
__global__ void isp_kernel(const uint16_t *__restrict__ raw, size_t raw_pitch16,
                           uint8_t *__restrict__ outY, size_t pitchY,
                           uint8_t *__restrict__ outUV, size_t pitchUV,
                           const uint8_t *__restrict__ tone_lut,
                           int W, int H, int do_dither, int flip180, unsigned frame_no,
                           int do_stats, AwbStats *stats)
{
    const int qx = blockIdx.x * blockDim.x + threadIdx.x;  /* quad coords */
    const int qy = blockIdx.y * blockDim.y + threadIdx.y;
    if (qx >= W / 2 || qy >= H / 2) return;
    const int x0 = qx * 2, y0 = qy * 2;

    float r8[4], g8[4], b8[4];
    float lin_sum[3] = {0, 0, 0};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int x = x0 + (i & 1), y = y0 + (i >> 1);
        float rgb[3];
        demosaic_rggb(raw, raw_pitch16, x, y, W, H, rgb);
        lin_sum[0] += rgb[0]; lin_sum[1] += rgb[1]; lin_sum[2] += rgb[2];
        float3 m = apply_mat(rgb);
        /* Highlight desaturation (NV12 path only; the RGB golden path stays
         * math-identical to the Python reference): sensor-clipped pixels
         * carry no color information, but the large WB gains (r~3.0) peg
         * R/B at the clamp while the CCM pulls G below it -> magenta
         * windows. Blend toward neutral as the RAW input approaches clip
         * (t ramps 0..1 over the top ~5% of the input range).             */
        if (!OUT_RGB) {
            float maxin = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
            float t = (maxin - 62259.0f) * (1.0f / (65535.0f - 62259.0f));
            if (t > 0.0f) {
                t = fminf(t, 1.0f);
                float mo = fmaxf(m.x, fmaxf(m.y, m.z));
                m.x += t * (mo - m.x);
                m.y += t * (mo - m.y);
                m.z += t * (mo - m.z);
            }
        }
        r8[i] = (float)tone_lut[(int)m.x];
        g8[i] = (float)tone_lut[(int)m.y];
        b8[i] = (float)tone_lut[(int)m.z];
    }

    if (OUT_RGB) {
        /* golden-test path: interleaved RGB8, no dither (determinism) */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int x = x0 + (i & 1), y = y0 + (i >> 1);
            uint8_t *p = outY + (size_t)y * pitchY + 3 * x;
            p[0] = (uint8_t)r8[i]; p[1] = (uint8_t)g8[i]; p[2] = (uint8_t)b8[i];
        }
    } else {
        /* BT.601 limited range; c_yuv = {yr,yg,yb, ur,ug,ub, vr,vg,vb, y_off, c_off, unused} */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int x = x0 + (i & 1), y = y0 + (i >> 1);
            float Y = c_yuv[0]*r8[i] + c_yuv[1]*g8[i] + c_yuv[2]*b8[i] + c_yuv[9];
            if (do_dither) Y += dither_of(x, y, frame_no);
            const int ox = flip180 ? (W - 1 - x) : x;
            const int oy = flip180 ? (H - 1 - y) : y;
            outY[(size_t)oy * pitchY + ox] = (uint8_t)clampi((int)lrintf(Y), 0, 255);
        }
        float ar = 0.25f*(r8[0]+r8[1]+r8[2]+r8[3]);
        float ag = 0.25f*(g8[0]+g8[1]+g8[2]+g8[3]);
        float ab = 0.25f*(b8[0]+b8[1]+b8[2]+b8[3]);
        float U = c_yuv[3]*ar + c_yuv[4]*ag + c_yuv[5]*ab + c_yuv[10];
        float V = c_yuv[6]*ar + c_yuv[7]*ag + c_yuv[8]*ab + c_yuv[10];
        if (do_dither) {
            U += dither_of(qx, qy, frame_no ^ 0x5555u);
            V += dither_of(qx, qy, frame_no ^ 0xAAAAu);
        }
        const int oqx = flip180 ? (W / 2 - 1 - qx) : qx;
        const int oqy = flip180 ? (H / 2 - 1 - qy) : qy;
        uint8_t *uv = outUV + (size_t)oqy * pitchUV + 2 * oqx;
        uv[0] = (uint8_t)clampi((int)lrintf(U), 0, 255);
        uv[1] = (uint8_t)clampi((int)lrintf(V), 0, 255);
    }

    /* AWB statistics: quad-mean linear RGB, exclusion of near-black/clip.
     * Matches the validated Python: central 2/3 window, luma in (0.016, 0.95). */
    if (do_stats) {
        if (x0 > W / 6 && x0 < W - W / 6 && y0 > H / 6 && y0 < H - H / 6) {
            float lr = lin_sum[0] * 0.25f, lg = lin_sum[1] * 0.25f, lb = lin_sum[2] * 0.25f;
            float lum = (lr + lg + lb) * (1.0f / 3.0f / 65535.0f);
            if (lum > 0.016f && lum < 0.95f) {
                atomicAdd(&stats->sum[0], lr);
                atomicAdd(&stats->sum[1], lg);
                atomicAdd(&stats->sum[2], lb);
                atomicAdd(&stats->cnt, 1.0f);
            }
        }
    }
}

/* ================= host side ================= */

struct IspCore {
    IspTuning tuning;
    IspParams params;
    int W, H;
    cudaStream_t stream;
    uint8_t *d_tone_lut;      /* 65536 entries */
    AwbStats *d_stats, *h_stats;
    double awb_ct;
    double wb_gains[3];
    unsigned stats_inflight;
    unsigned frame_count;
};

void isp_params_defaults(IspParams *p)
{
    memset(p, 0, sizeof(*p));
    p->preset = TONE_TUNING;
    p->contrast = 1.0; p->brightness = 0.0; p->saturation = 1.0;
    p->digital_gain = 1.0; p->dither = 1; p->black_offset = 0;
    p->knee_point = 1.0; p->knee_strength = 0.0;
    p->awb = AWB_AUTO; p->awb_ct = 4560.0;
}

/* tone curve value for linear input t in [0,1] under the active preset */
static double tone_curve(const IspTuning *tu, const IspParams *p, double t)
{
    double v;
    switch (p->preset) {
    case TONE_LINEAR: v = t; break;
    case TONE_SRGB:
        v = (t <= 0.0031308) ? 12.92 * t : 1.055 * pow(t, 1.0 / 2.4) - 0.055; break;
    case TONE_REC709:
        v = (t < 0.018) ? 4.5 * t : 1.099 * pow(t, 0.45) - 0.099; break;
    default: { /* TONE_TUNING: piecewise-linear interp of the RPi curve */
        const int n = tu->n_gamma;
        if (t <= tu->gamma_x[0]) { v = tu->gamma_y[0]; break; }
        if (t >= tu->gamma_x[n-1]) { v = tu->gamma_y[n-1]; break; }
        int i = 1; while (i < n - 1 && tu->gamma_x[i] < t) i++;
        double x0 = tu->gamma_x[i-1], x1 = tu->gamma_x[i];
        double y0 = tu->gamma_y[i-1], y1 = tu->gamma_y[i];
        v = y0 + (y1 - y0) * (t - x0) / (x1 - x0 + 1e-12);
    } }
    return v;
}

/* Build the 65536-entry tone LUT: knee (linear domain) -> curve ->
 * contrast (S about mid) -> brightness -> 8-bit. */
static void build_tone_lut(const IspTuning *tu, const IspParams *p, uint8_t *lut)
{
    /* optional custom LUT file: 1024 lines of 0..255 (or 0..1 floats) */
    if (p->lut_file[0]) {
        FILE *f = fopen(p->lut_file, "r");
        if (f) {
            double v[1024]; int n = 0; double x;
            while (n < 1024 && fscanf(f, "%lf", &x) == 1) v[n++] = x;
            fclose(f);
            if (n == 1024) {
                double scale = 1.0;
                for (int i = 0; i < 1024; i++) if (v[i] > 1.5) { scale = 1.0 / 255.0; break; }
                for (int i = 0; i < 65536; i++) {
                    double idx = i * (1023.0 / 65535.0);
                    int i0 = (int)idx; int i1 = i0 < 1023 ? i0 + 1 : 1023;
                    double fr = idx - i0;
                    double y = (v[i0] * (1 - fr) + v[i1] * fr) * scale;
                    lut[i] = (uint8_t)fmin(fmax(y * 255.0 + 0.5, 0.0), 255.0);
                }
                return;
            }
            fprintf(stderr, "nvimx296: lut-file %s malformed (%d values, want 1024) - ignored\n",
                    p->lut_file, n);
        } else {
            fprintf(stderr, "nvimx296: cannot open lut-file %s - ignored\n", p->lut_file);
        }
    }

    const double kp = p->knee_point, ks = p->knee_strength;
    for (int i = 0; i < 65536; i++) {
        double t = i / 65535.0;
        /* highlight knee: soft-compress above kp in the LINEAR domain */
        if (ks > 0.0 && kp < 1.0 && t > kp) {
            double over = (t - kp) / (1.0 - kp);            /* 0..1 */
            double soft = kp + (1.0 - kp) * (over / (1.0 + ks * over)) * (1.0 + ks);
            /* normalized so soft(1)=1; blends toward hard clip as ks->0 */
            t = fmin(soft, 1.0);
        }
        double v = tone_curve(tu, p, t);
        /* contrast: S-curve pivot at the curve's mid response */
        if (p->contrast != 1.0) {
            const double pivot = 0.5;
            v = pivot + (v - pivot) * p->contrast;
        }
        v += p->brightness;
        lut[i] = (uint8_t)fmin(fmax(v * 255.0 + 0.5, 0.0), 255.0);
    }
}

/* lin LUT: (p10 - black - offset) * 65535/(1023 - black), clamped */
static void build_lin_lut(const IspTuning *tu, const IspParams *p, uint16_t *lut)
{
    const double bl = tu->black_level_16bit / 64.0 + p->black_offset;
    const double scale = 65535.0 / (1023.0 - bl > 1.0 ? 1023.0 - bl : 1.0);
    for (int i = 0; i < 1024; i++) {
        double v = (i - bl) * scale;
        lut[i] = (uint16_t)fmin(fmax(v, 0.0), 65535.0);
    }
}

static void select_ccm(const IspTuning *tu, double ct, double m[9])
{
    int best = 0; double bd = 1e18;
    for (int i = 0; i < tu->n_ccms; i++) {
        double d = fabs(tu->ccm_ct[i] - ct);
        if (d < bd) { bd = d; best = i; }
    }
    memcpy(m, tu->ccm[best], sizeof(double) * 9);
}

static void wb_from_curve(const IspTuning *tu, double ct, double gains[3])
{
    int best = 0; double bd = 1e18;
    for (int i = 0; i < tu->n_ct; i++) {
        double d = fabs(tu->ct_curve[i][0] - ct);
        if (d < bd) { bd = d; best = i; }
    }
    gains[0] = 1.0 / tu->ct_curve[best][1];
    gains[1] = 1.0;
    gains[2] = 1.0 / tu->ct_curve[best][2];
}

static int upload_state(IspCore *c)
{
    /* combined matrix = dgain * CCM @ diag(wb) */
    double ccm[9]; select_ccm(&c->tuning, c->awb_ct, ccm);
    float m[9];
    for (int r = 0; r < 3; r++)
        for (int k = 0; k < 3; k++)
            m[r * 3 + k] = (float)(c->params.digital_gain * ccm[r * 3 + k] * c->wb_gains[k]);
    CUCHK(cudaMemcpyToSymbolAsync(c_mat, m, sizeof(m), 0, cudaMemcpyHostToDevice, c->stream));

    uint16_t lin[1024]; build_lin_lut(&c->tuning, &c->params, lin);
    CUCHK(cudaMemcpyToSymbolAsync(c_lin_lut, lin, sizeof(lin), 0, cudaMemcpyHostToDevice, c->stream));

    uint8_t *tone = (uint8_t *)malloc(65536);
    build_tone_lut(&c->tuning, &c->params, tone);
    CUCHK(cudaMemcpyAsync(c->d_tone_lut, tone, 65536, cudaMemcpyHostToDevice, c->stream));
    CUCHK(cudaStreamSynchronize(c->stream));
    free(tone);

    /* BT.601 limited-range RGB8->YUV, saturation folded into chroma rows */
    const double s = c->params.saturation;
    float yuv[12] = {
        (float)(65.738/256), (float)(129.057/256), (float)(25.064/256),
        (float)(s * -37.945/256), (float)(s * -74.494/256), (float)(s * 112.439/256),
        (float)(s * 112.439/256), (float)(s * -94.154/256), (float)(s * -18.285/256),
        16.0f, 128.0f, 0.0f
    };
    CUCHK(cudaMemcpyToSymbol(c_yuv, yuv, sizeof(yuv)));
    return 0;
}

IspCore *isp_core_create(const IspTuning *t, const IspParams *p,
                         int width, int height, cudaStream_t stream)
{
    IspCore *c = (IspCore *)calloc(1, sizeof(IspCore));
    if (!c) return NULL;
    c->tuning = *t; c->params = *p; c->W = width; c->H = height; c->stream = stream;
    c->awb_ct = p->awb_ct;
    wb_from_curve(t, c->awb_ct, c->wb_gains);
    if (p->awb == AWB_OFF) { c->wb_gains[0] = c->wb_gains[1] = c->wb_gains[2] = 1.0; }
    if (cudaMalloc(&c->d_tone_lut, 65536) != cudaSuccess) { free(c); return NULL; }
    if (cudaMalloc(&c->d_stats, sizeof(AwbStats)) != cudaSuccess ||
        cudaMallocHost(&c->h_stats, sizeof(AwbStats)) != cudaSuccess) {
        isp_core_destroy(c); return NULL;
    }
    cudaMemset(c->d_stats, 0, sizeof(AwbStats));
    if (upload_state(c) != 0) { isp_core_destroy(c); return NULL; }
    return c;
}

void isp_core_destroy(IspCore *c)
{
    if (!c) return;
    if (c->d_tone_lut) cudaFree(c->d_tone_lut);
    if (c->d_stats) cudaFree(c->d_stats);
    if (c->h_stats) cudaFreeHost(c->h_stats);
    free(c);
}

int isp_core_update_params(IspCore *c, const IspParams *p)
{
    AwbMode prev = c->params.awb;
    c->params = *p;
    if (p->awb != prev) {
        if (p->awb == AWB_OFF) {
            c->wb_gains[0] = c->wb_gains[1] = c->wb_gains[2] = 1.0;
        } else {
            c->awb_ct = p->awb_ct;
            wb_from_curve(&c->tuning, c->awb_ct, c->wb_gains);
        }
    } else if (p->awb == AWB_TUNING && p->awb_ct != c->awb_ct) {
        c->awb_ct = p->awb_ct;
        wb_from_curve(&c->tuning, c->awb_ct, c->wb_gains);
    }
    return upload_state(c);
}

void isp_core_get_awb(IspCore *c, double *ct, double gains[3])
{
    if (ct) *ct = c->awb_ct;
    if (gains) memcpy(gains, c->wb_gains, sizeof(c->wb_gains));
}

#define AWB_PERIOD 15  /* run stats every Nth frame */

static int launch(IspCore *c, bool out_rgb,
                  const uint16_t *d_raw, size_t raw_pitch,
                  uint8_t *dY, size_t pY, uint8_t *dUV, size_t pUV, unsigned fno)
{
    dim3 blk(16, 16);
    dim3 grd((c->W / 2 + blk.x - 1) / blk.x, (c->H / 2 + blk.y - 1) / blk.y);
    int stats = (!out_rgb && c->params.awb == AWB_AUTO &&
                 (fno % AWB_PERIOD) == 0 && !c->stats_inflight);
    if (stats) CUCHK(cudaMemsetAsync(c->d_stats, 0, sizeof(AwbStats), c->stream));
    if (out_rgb)
        isp_kernel<true><<<grd, blk, 0, c->stream>>>(
            d_raw, raw_pitch / 2, dY, pY, NULL, 0, c->d_tone_lut,
            c->W, c->H, 0, 0, fno, 0, NULL);
    else
        isp_kernel<false><<<grd, blk, 0, c->stream>>>(
            d_raw, raw_pitch / 2, dY, pY, dUV, pUV, c->d_tone_lut,
            c->W, c->H, c->params.dither, c->params.flip180, fno, stats, c->d_stats);
    CUCHK(cudaGetLastError());
    if (stats) {
        CUCHK(cudaMemcpyAsync(c->h_stats, c->d_stats, sizeof(AwbStats),
                              cudaMemcpyDeviceToHost, c->stream));
        c->stats_inflight = 1;
    }
    return 0;
}

int isp_core_process_nv12(IspCore *c, const uint16_t *d_raw, size_t raw_pitch,
                          uint8_t *dY, size_t pY, uint8_t *dUV, size_t pUV,
                          unsigned fno)
{ c->frame_count = fno; return launch(c, false, d_raw, raw_pitch, dY, pY, dUV, pUV, fno); }

int isp_core_process_rgb(IspCore *c, const uint16_t *d_raw, size_t raw_pitch,
                         uint8_t *d_rgb, size_t pitch_rgb, unsigned fno)
{ return launch(c, true, d_raw, raw_pitch, d_rgb, pitch_rgb, NULL, 0, fno); }

int isp_core_awb_tick(IspCore *c)
{
    if (!c->stats_inflight) return 0;
    /* stream has been synchronized by the caller after process(); the
     * readback is complete. */
    c->stats_inflight = 0;
    AwbStats *s = c->h_stats;
    if (s->cnt < 500.0f) return 0;
    double r = s->sum[0] / s->cnt, g = s->sum[1] / s->cnt, b = s->sum[2] / s->cnt;
    if (r <= 0 || g <= 0 || b <= 0) return 0;
    double rr = r / g, br = b / g;
    /* nearest illuminant on the calibrated curve (ratio space) */
    int best = 0; double bd = 1e18;
    for (int i = 0; i < c->tuning.n_ct; i++) {
        double d = (c->tuning.ct_curve[i][1] - rr) * (c->tuning.ct_curve[i][1] - rr)
                 + (c->tuning.ct_curve[i][2] - br) * (c->tuning.ct_curve[i][2] - br);
        if (d < bd) { bd = d; best = i; }
    }
    const double a = 0.25;  /* EMA */
    double tgt[3] = { 1.0 / c->tuning.ct_curve[best][1], 1.0,
                      1.0 / c->tuning.ct_curve[best][2] };
    for (int k = 0; k < 3; k++)
        c->wb_gains[k] = (1 - a) * c->wb_gains[k] + a * tgt[k];
    c->awb_ct = (1 - a) * c->awb_ct + a * c->tuning.ct_curve[best][0];
    return upload_state(c) == 0 ? 1 : 0;
}
