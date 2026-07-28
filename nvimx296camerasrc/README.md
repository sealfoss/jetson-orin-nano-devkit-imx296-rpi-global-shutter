# nvimx296camerasrc

GStreamer source element for the Raspberry Pi Global Shutter Camera (Sony
IMX296) on Jetson Orin: raw V4L2 capture + fused CUDA ISP using the real
RPi/libcamera tuning data, outputting NV12 in `memory:NVMM` buffers.
**Bypasses Argus entirely** — no AE hunting, no TNR, no untuned color.

```
V4L2 RG10 (10-bit raw) → HtoD → one fused CUDA kernel:
  black-level LUT → bilinear debayer (linear) → WB·CCM·digital-gain 3×3
  → tone LUT (preset/contrast/brightness/knee) → BT.601 NV12 (+dither)
→ NvBufSurface (NVMM) → downstream (nv3dsink / nvv4l2h265enc / nvjpegenc …)
```

## Build (native, on the Jetson)

```bash
cd nvimx296camerasrc && mkdir -p build && cd build
cmake .. && make -j8
export GST_PLUGIN_PATH=$PWD
gst-inspect-1.0 nvimx296camerasrc
```

Requires: JetPack 6 (CUDA 12.x), gstreamer dev headers,
`/usr/src/jetson_multimedia_api` (stock), the sensor driver + overlay from
this repo, and the tuning JSON (`doc/external_sources/imx296_16mm.json`).

## Use

```bash
# live view (display attached):
gst-launch-1.0 nvimx296camerasrc exposure=8333 gain=60 ! \
  'video/x-raw(memory:NVMM),width=1456,height=1088,framerate=60/1' ! nv3dsink

# 720p @ 90 fps H.265 recording:
gst-launch-1.0 -e nvimx296camerasrc exposure=8333 gain=80 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=90/1' ! \
  nvv4l2h265enc bitrate=20000000 ! h265parse ! matroskamux ! filesink location=clip.mkv

# tone controls, all runtime-mutable:
... nvimx296camerasrc contrast=1.2 saturation=1.3 tone-preset=tuning \
    knee-point=0.85 knee-strength=0.5 awb=auto ...
```

## Properties

| name | range (default) | notes |
|---|---|---|
| `device` | (`/dev/video0`) | |
| `tuning-file` | path | RPi imx296 tuning JSON |
| `exposure` | 15–15699 µs (8333) | 8333 = mains-ripple immune |
| `gain` | 0–480 dB×10 (60) | sensor analog gain |
| `tone-preset` | tuning/srgb/rec709/linear | `linear` for CV consumers |
| `contrast` | 0–2 (1) | |
| `brightness` | −1–1 (0) | |
| `saturation` | 0–2 (1) | folded into NV12 chroma matrix |
| `digital-gain` | 0.25–4 (1) | linear, pre-curve |
| `dither` | bool (true) | blue-noise at 8-bit quantize |
| `black-offset` | −32–32 (0) | manual pedestal trim (BLKLEVELAUTO is disabled in the driver) |
| `knee-point` / `knee-strength` | 0.5–1 (1=off) / 0–1 (0) | highlight rolloff |
| `tone-lut-file` | path | 1024-entry curve override |
| `awb` | auto/tuning/off | auto = grey-world constrained to the calibrated CT curve |
| `awb-ct` | 2000–10000 (4560) | for `awb=tuning` |
| `flip` | bool (false) | rotate output 180° (upside-down mount) |

Highlight handling: sensor-clipped pixels are automatically desaturated
toward white over the top ~5% of the raw range (prevents WB-gain magenta in
blown highlights; independent of the `knee-*` shoulder controls).

## Golden test

`imx296_kernel_test raw.bin tuning.json out.rgb [W H STRIDE] [ct]` runs the
exact element kernel on a canned raw capture and dumps RGB8; compare against
`scripts/imx296_isp_pipeline.py --input raw.bin --awb tuning --ct <ct>`.
Also prints kernel-only timing for the production NV12 path.

## Scope notes

- **Fully zero-copy end to end**: the VI DMAs raw frames straight into
  EGL/CUDA-mapped NvBufSurface dmabufs (V4L2_MEMORY_DMABUF; the surface
  pitch is imposed on the VI via `preferred_stride`), the kernel reads them
  in place, and output NVMM surfaces go downstream without copies. If the
  dmabuf negotiation fails the element falls back to MMAP + pinned HtoD
  automatically (`zero-copy=false` forces the fallback for comparison).
- NV12 8-bit out only (P010 dead-ends downstream: nvivafilter etc. are 8-bit).
- Single element instance per sensor; 1456×1088 and 1280×720 modes.
