# IMX296 Driver for Jetson Orin (hardened fork)

A Linux V4L2 sensor driver and device-tree overlays that bring the Sony
IMX296 global-shutter sensor (the **Raspberry Pi Global Shutter Camera**) to
**NVIDIA Jetson Orin** devkits, using NVIDIA's `tegracam` framework.

This is a **fork** of Jonathan Péclat's
[jetson-orin-nano-devkit-imx296-rpi-global-shutter](https://github.com/peclatj/jetson-orin-nano-devkit-imx296-rpi-global-shutter),
which did the original bring-up (and whose register documentation in `doc/`
remains the best public reference for this sensor). On top of it, this fork
fixes several functional defects found by adversarial review and hardware
forensics, adds dual-camera and 90 fps support, and pairs the driver with a
full custom-ISP path that bypasses Argus entirely.

> **Status: working.** Probes, streams, and records on Jetson Orin NX 16GB
> (devkit carrier, JetPack 6 / L4T r36.5.0) in daily use; original upstream
> was validated on Orin Nano (JetPack 6.2.2). The upstream README's two big
> known issues — the black line on every frame and dull/flickering images —
> are fixed or root-caused here (see below).

## What this fork changes vs upstream

| Change | Why |
|---|---|
| **Exposure control fixed** | Upstream subtracted the 14 µs readout offset scaled ×1,000,000 (= 14 *seconds*) — the math wrapped and pinned SHS1 at max: the V4L2/Argus exposure control silently did nothing. Now correct and verified monotonic on hardware. |
| **Black horizontal line fixed** | Two register deltas vs the RPi *production* driver: init byte `0x30af = 0x0b` (RPi's Fast-Trigger/MIPI-FE fix, on Sony's advice) and the missing `MIPIC_AREA3W (0x4182) = height` write. Verified gone on real captures. |
| **Shadow/line flicker fixed** | The sensor's automatic black-level servo (`BLKLEVELAUTO`) oscillates ±1.5 counts @ 7–9 Hz (±35 @ 30 dB gain) — proven with capped-lens dark frames and a mid-stream register A/B. Disabled in favor of the fixed level the tuning data assumes (trade-off: no thermal-drift compensation; a manual trim exists in the companion ISP element). |
| **Gain latches at frame boundary** | `GAINDLY` 1-frame mode (the RPi production value) so gain+exposure steps land atomically on one frame. |
| **1280×720 @ 90 fps mode (mode1)** | Centered ROI crop, VMAX 750 → exactly 90.0 fps. Includes making the driver's VMAX floor mode-relative — without that, the frame-rate control silently clamps the new mode back to ~60 fps. |
| **Dual-camera overlay** | `tegra234-p3767-camera-p3768-imx296-dual.dtbo` (both CSI connectors at once, `video0` + `video1`), modeled on NVIDIA's imx477-dual overlay. |
| **Overlay defects removed** | Upstream's PWDN gpio-hogs targeted a Tegra210 address that doesn't exist on Orin (silently inert — and dangerous to "fix" in place); phantom disable nodes; a `channel@1`/`reg=<0>` contradiction in the -C overlay. |
| **`#define DEBUG` off by default** | Upstream shipped with full register dumps plus a 1 s sleep inside every stream start. |
| **ISP script rewritten** (`scripts/imx296_isp_pipeline.py`) | Upstream's didn't run (wrong tuning path). Now: offline/live raw → color pipeline with curve-constrained auto-AWB, argparse CLI, both sensor modes. |
| Register provenance comments | Every deviating byte cites its source (mainline vs RPi tree vs measured). |

Every change was merged with `--no-ff`, one branch per fix — `git log` is
the changelog, and any fix can be reverted as a single merge commit.

## Repository layout

```
source/
  nvidia-oot/drivers/media/i2c/
    nv_imx296.c              Sensor driver (tegracam)
    imx296_mode_tbls.h       Register tables: mode0 1456x1088@60, mode1 1280x720@90
    Makefile                 Adds nv_imx296.o to the i2c module list
  hardware/nvidia/t23x/nv-public/overlay/
    tegra234-p3767-camera-p3768-imx296-A.dts     CAM0 (J20) single
    tegra234-p3767-camera-p3768-imx296-C.dts     CAM1 (J21) single
    tegra234-p3767-camera-p3768-imx296-dual.dts  both connectors
    Makefile                 Adds the three .dtbo targets

scripts/                     Build/deploy helpers (upstream's) + the ISP pipeline
doc/                         Register reference (typ/pdf) + mirrored sources
```

The files under `source/` mirror the L4T `Linux_for_Tegra/source` layout and
copy straight on top of it.

**Companion project:** the recommended way to *consume* this driver is
[`nvimx296camerasrc`][nvimx296camerasrc] — a GStreamer source element with a
fused CUDA ISP that uses the real RPi/libcamera tuning data and replaces
`nvarguscamerasrc`/Argus entirely: no AE hunting, no TNR, correct color,
zero-copy from sensor DMA to `memory:NVMM` NV12 at 60/90 fps. It began life
in this repo (see git history through the `feat/cuda-isp` /
`feat/zero-copy-capture` merges) and is maintained as its own CMake project.

## Build

Against an extracted JetPack 6 `Linux_for_Tegra/source` tree (versions must
match the target's L4T exactly — check `cat /etc/nv_tegra_release`):

```bash
cp -r source/* /path/to/Linux_for_Tegra/source/
cd /path/to/Linux_for_Tegra/source
export KERNEL_HEADERS=/usr/src/linux-headers-$(uname -r)-ubuntu22.04_aarch64/3rdparty/canonical/linux-jammy/kernel-source
make modules   # -> nvidia-oot/drivers/media/i2c/nv_imx296.ko
make dtbs      # -> kernel-devicetree/generic-dts/dtbs/tegra234-p3767-camera-p3768-imx296-{A,C,dual}.dtbo
```

(Native build on the Jetson works with the stock `linux-headers` package, as
above; cross-building from x86 works with the upstream `scripts/build_*.sh`
after adjusting the hardcoded paths.)

## Install on the target

1. Copy the `.dtbo`(s) to `/boot/` and the `.ko` somewhere convenient.
2. Add a **non-default** extlinux label (keep your known-good default
   bootable — a bad camera boot then costs one power-cycle, nothing more):

```
LABEL imx296
    MENU LABEL IMX296 GS camera
    LINUX /boot/Image
    FDT /boot/dtb/<your-base-dtb>.dtb
    INITRD /boot/initrd
    APPEND <copy the APPEND line of your working label>
    OVERLAYS /boot/tegra234-p3767-camera-p3768-imx296-A.dtbo
```

Use `...-C.dtbo` for CAM1 or `...-dual.dtbo` for both. Do **not** co-apply
the two single overlays (they share `video0`).

3. Reboot, pick the label at the boot menu, and load the driver manually:

```bash
sudo insmod nv_imx296.ko     # keep it manual until proven on your setup;
                             # auto-loading an unproven camera driver in
                             # modules-load.d is how boards get bricked
sudo dmesg | grep imx296
```

Expected probe:

```
imx296 9-001a: probing IMX296 sensor
imx296 9-001a: IMX296LQ (color) detected (sensor_info=0x4a00)
tegra-camrtc-capture-vi tegra-capture-vi: subdev imx296 9-001a bound
imx296 9-001a: IMX296LQ sensor detected and registered
```

## Capturing

**Raw V4L2** (both modes; Orin's VI needs a 64-byte-aligned stride —
`preferred_stride` below):

```bash
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1456,height=1088,pixelformat=RG10 \
         -c preferred_stride=2944
v4l2-ctl -d /dev/video0 -c exposure=8333,gain=100    # us; dB*10
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-skip=4 \
         --stream-to=frame.raw
python3 scripts/imx296_isp_pipeline.py --input frame.raw   # -> color PNG (auto AWB)
```

**Best quality / live** — use the companion
[`nvimx296camerasrc`][nvimx296camerasrc] element (see above).

**Via Argus** (`nvarguscamerasrc`) — works, but Argus has no tuning profile
for this sensor: colors are approximate and the untuned auto-exposure loop
*hunts* (~25% brightness oscillation measured). If you must use it, pin
everything:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-mode=0 tnr-mode=0 ee-mode=0 \
    aelock=true awblock=true aeantibanding=0 \
    exposuretimerange="8333000 8333000" gainrange="60 60" \
    ispdigitalgainrange="1 1" \
  ! 'video/x-raw(memory:NVMM),width=1456,height=1088,framerate=60/1' ! nv3dsink
```

(8.333 ms is the only mains-flicker-immune exposure at any frame rate; gain
is dB×10, 0–480, halve exposure ⇒ +60 gain.)

## Known issues / limitations

- **Argus colors remain untuned** (NVIDIA provides no path to load a
  third-party tuning file) — by design unfixable in Argus; solved properly
  by the companion CUDA ISP element instead.
- **CFA phase mystery**: raw-domain site statistics look G-first while the
  `RG10` fourcc claims RGGB; the demosaic mapping used everywhere here is
  empirically color-correct, but the discrepancy is unexplained. Affects
  only code doing mosaic-domain math.
- Do not `rmmod` the driver while an Argus client is running/tearing down —
  NVIDIA's camera stack races a use-after-free (observed kernel panic).
  Stop `nvargus-daemon` first.
- Monochrome variant (IMX296LL) is detected but has never been
  hardware-tested; no EEPROM/OTP/HDR support.
- Dual-overlay operation is code-complete and boots, but simultaneous
  two-camera streaming awaits second-camera hardware validation.
- With `BLKLEVELAUTO` disabled (flicker fix), slow thermal black-level drift
  is uncompensated — if you see shadow lift in long sessions, use the ISP
  element's `black-offset` property.

## Credits

- **Jonathan Péclat** — original bring-up, driver, overlays, and register
  documentation this fork stands on.
- Mainline `drivers/media/i2c/imx296.c` (Laurent Pinchart) and the Raspberry
  Pi kernel/libcamera projects — register sequences, tuning data, and the
  production reference against which the fixes here were verified.
- NVIDIA's `nv_imx185`/`gst-nvv4l2camera` sources — tegracam and NVMM
  conventions.

## License

GPL-2.0, see [LICENSE](LICENSE).

<!-- Companion-repo link: update this ONE definition when the
     nvimx296camerasrc repository is published. -->
[nvimx296camerasrc]: https://github.com/sealfoss/GStreamer-NV-IMX286-Camera-Source