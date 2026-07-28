/* SPDX-License-Identifier: GPL-2.0-only
 * Golden test for the fused ISP kernel: load a raw RG10 capture (as written
 * by v4l2-ctl --stream-to, including stride padding), run the SAME kernel
 * the element uses, dump interleaved RGB8. Compared offline against the
 * validated Python still pipeline on identical input.
 *
 * usage: imx296_kernel_test <raw> <tuning.json> <out.rgb> [W H STRIDE] [ct]
 *   default geometry 1456 1088 2944; ct fixes AWB (awb=tuning) for determinism
 */
#include "imx296_isp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s raw tuning.json out.rgb [W H STRIDE] [ct]\n", argv[0]);
        return 2;
    }
    const char *rawp = argv[1], *tunp = argv[2], *outp = argv[3];
    int W = argc > 6 ? atoi(argv[4]) : 1456;
    int H = argc > 6 ? atoi(argv[5]) : 1088;
    int STRIDE = argc > 6 ? atoi(argv[6]) : 2944;
    double ct = argc > 7 ? atof(argv[7]) : 4560.0;

    IspTuning t;
    if (isp_tuning_load(tunp, &t) != 0) return 1;
    printf("tuning: black=%.0f ccms=%d ct_pts=%d gamma_pts=%d\n",
           t.black_level_16bit, t.n_ccms, t.n_ct, t.n_gamma);

    FILE *f = fopen(rawp, "rb");
    if (!f) { perror(rawp); return 1; }
    size_t fsz = (size_t)STRIDE * H;
    uint8_t *host = (uint8_t *)malloc(fsz);
    if (fread(host, 1, fsz, f) != fsz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint16_t *d_raw; size_t d_pitch;
    cudaMallocPitch((void **)&d_raw, &d_pitch, W * 2, H);
    cudaMemcpy2D(d_raw, d_pitch, host, STRIDE, W * 2, H, cudaMemcpyHostToDevice);

    uint8_t *d_rgb; size_t rgb_pitch;
    cudaMallocPitch((void **)&d_rgb, &rgb_pitch, W * 3, H);

    IspParams p;
    isp_params_defaults(&p);
    p.awb = AWB_TUNING;      /* deterministic: fixed CT, like python --awb tuning */
    p.awb_ct = ct;
    p.dither = 0;

    IspCore *core = isp_core_create(&t, &p, W, H, stream);
    if (!core) { fprintf(stderr, "core create failed\n"); return 1; }

    double gains[3]; double cur_ct;
    isp_core_get_awb(core, &cur_ct, gains);
    printf("awb: ct=%.0f gains=[%.3f 1.0 %.3f]\n", cur_ct, gains[0], gains[2]);

    if (isp_core_process_rgb(core, d_raw, d_pitch, d_rgb, rgb_pitch, 0) != 0) return 1;
    cudaError_t e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(e)); return 1; }

    uint8_t *out = (uint8_t *)malloc((size_t)W * H * 3);
    cudaMemcpy2D(out, W * 3, d_rgb, rgb_pitch, W * 3, H, cudaMemcpyDeviceToHost);
    f = fopen(outp, "wb");
    fwrite(out, 1, (size_t)W * H * 3, f);
    fclose(f);
    printf("wrote %s (%dx%d RGB8)\n", outp, W, H);

    /* timing: 100 iterations of the NV12 path (the production path) */
    uint8_t *dY, *dUV; size_t pY, pUV;
    cudaMallocPitch((void **)&dY, &pY, W, H);
    cudaMallocPitch((void **)&dUV, &pUV, W, H / 2);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    isp_core_process_nv12(core, d_raw, d_pitch, dY, pY, dUV, pUV, 1); /* warm */
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < 100; i++)
        isp_core_process_nv12(core, d_raw, d_pitch, dY, pY, dUV, pUV, i + 2);
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("NV12 kernel: %.3f ms/frame (%.0f fps equivalent)\n",
           ms / 100.0, 100000.0 / ms);
    isp_core_destroy(core);
    return 0;
}
