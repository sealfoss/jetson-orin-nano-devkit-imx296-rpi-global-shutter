/* SPDX-License-Identifier: GPL-2.0-only
 * Fused CUDA ISP for the IMX296: interface between the GStreamer element,
 * the golden-test tool, and the kernel. All tone/color knobs pre-bake on
 * the CPU into one 3x3 matrix + two LUTs; the kernel never changes shape.
 */
#ifndef IMX296_ISP_H
#define IMX296_ISP_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- tuning data parsed from the RPi imx296 JSON (tuning.cpp) ---- */
#define ISP_MAX_CCMS 16
#define ISP_MAX_CT_POINTS 16
#define ISP_GAMMA_MAX_POINTS 64

typedef struct {
    double black_level_16bit;              /* e.g. 3840 (=60<<6) */
    int    n_ccms;
    double ccm_ct[ISP_MAX_CCMS];
    double ccm[ISP_MAX_CCMS][9];           /* row-major RGB matrices */
    int    n_ct;
    double ct_curve[ISP_MAX_CT_POINTS][3]; /* ct, r/g ratio, b/g ratio */
    int    n_gamma;
    double gamma_x[ISP_GAMMA_MAX_POINTS];  /* 0..1 */
    double gamma_y[ISP_GAMMA_MAX_POINTS];  /* 0..1 */
} IspTuning;

int isp_tuning_load(const char *json_path, IspTuning *out); /* 0 = ok */

/* ---- user-facing tone/color parameters (element properties) ---- */
typedef enum { TONE_TUNING = 0, TONE_SRGB, TONE_REC709, TONE_LINEAR } TonePreset;
typedef enum { AWB_AUTO = 0, AWB_TUNING, AWB_OFF } AwbMode;

typedef struct {
    TonePreset preset;
    double contrast;        /* 0..2, 1 = neutral            */
    double brightness;      /* -1..1, 0 = neutral           */
    double saturation;      /* 0..2, 1 = neutral            */
    double digital_gain;    /* 0.25..4                      */
    int    dither;          /* bool                         */
    int    black_offset;    /* -32..32 counts (10-bit)      */
    double knee_point;      /* 0.5..1.0; 1.0 = knee off     */
    double knee_strength;   /* 0..1                         */
    char   lut_file[512];   /* optional 1024-entry override */
    AwbMode awb;
    double awb_ct;          /* fixed CT for AWB_TUNING      */
    int    flip180;         /* rotate output 180 deg (mount) */
} IspParams;

void isp_params_defaults(IspParams *p);

/* ---- device-side state owned by the ISP core ---- */
typedef struct IspCore IspCore;

IspCore *isp_core_create(const IspTuning *t, const IspParams *p,
                         int width, int height, cudaStream_t stream);
void isp_core_destroy(IspCore *c);

/* Rebuild LUTs/matrix after a parameter change (cheap, async upload). */
int isp_core_update_params(IspCore *c, const IspParams *p);

/* Current AWB state (for HUD/debug). */
void isp_core_get_awb(IspCore *c, double *ct, double wb_gains[3]);

/* Process one frame.
 *  d_raw     : device pointer to RG10 frame (uint16 samples, VI 16-bit container)
 *  raw_pitch : bytes per input row
 *  dY, dUV   : device pointers to the NV12 output planes
 *  pitchY/UV : bytes per output row
 *  frame_no  : used for AWB cadence + dither decorrelation
 * Runs async on the stream; caller synchronizes.                        */
int isp_core_process_nv12(IspCore *c, const uint16_t *d_raw, size_t raw_pitch,
                          uint8_t *dY, size_t pitchY,
                          uint8_t *dUV, size_t pitchUV,
                          unsigned frame_no);

/* Test path: same math, RGB8 interleaved output (golden comparison). */
int isp_core_process_rgb(IspCore *c, const uint16_t *d_raw, size_t raw_pitch,
                         uint8_t *d_rgb, size_t pitch_rgb, unsigned frame_no);

/* Kick the async AWB statistics readback/update (call ~once per second
 * of frames; safe every frame). Returns 1 if gains changed. */
int isp_core_awb_tick(IspCore *c);

#ifdef __cplusplus
}
#endif
#endif
