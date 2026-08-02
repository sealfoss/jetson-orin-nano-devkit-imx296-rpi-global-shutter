# External trigger mode — how-to

The driver exposes the IMX296's external (XTR) **fast-trigger** mode as a
custom V4L2 boolean control, `trigger_mode`, on every camera's video node
and subdev. Armed, the sensor stops free-running: **each hardware pulse on
the XTR line produces exactly one frame, and the pulse width sets the
exposure time**. This is the mode the Raspberry Pi GS camera documents for
its own kernel driver (`CTRL0B = TRIGEN`, `LOWLAGTRG = FAST` — identical
registers, written in the identical stream-start slot).

Why you'd want it: drive both cameras from one shared pulse line and their
exposures are hardware-simultaneous — cross-camera frame association
becomes exact timestamp matching instead of "close". See
`dual-camera-klv-stream-design.md` §4.6/§9 in the parent project.

## Quick reference

| | free-run (default) | trigger mode |
|---|---|---|
| frame timing | `VMAX` / `frame_rate` control | one frame per XTR pulse |
| exposure | `SHS1` / `exposure` control | **XTR pulse width** |
| arm/disarm | — | `trigger_mode` control, latched at STREAMON |

## Using it

### 1. Check the control exists

```bash
v4l2-ctl -d /dev/video0 --list-ctrls | grep trigger_mode
#   trigger_mode 0x009a20c8 (bool) : default=0 value=0
```

Absent? Your installed module predates the trigger feature (or predates
the fix that registers the control early enough to survive the VI's
control snapshot — driver commit `90e86f2`). Rebuild + reinstall
(`scripts/build_modules.sh`, `scripts/install.sh`) and
`rmmod nv_imx296 && modprobe nv_imx296`.

### 2. Arm before streaming

The driver programs the sensor registers **at stream start** (the same
place the RPi driver does; the sensor state machine is not switched
mid-stream). Set the control first, then start the stream:

```bash
v4l2-ctl -d /dev/video0 -c trigger_mode=1
v4l2-ctl -d /dev/video1 -c trigger_mode=1        # both, for synced capture
# now start capture (v4l2-ctl / nvimx296camerasrc / your app)
```

Setting the control while already streaming takes effect at the *next*
stream start — restart the stream.

The control value **persists per device** until changed or module reload.
Return to free-run with `trigger_mode=0` (the registers themselves are
also written back to 0 on every free-run stream start, so a stale
triggered session can never contaminate a normal run).

### 3. Verify it latched

```bash
sudo dmesg | grep trigger
# imx296 9-001a: external trigger mode: frames follow XTR pulses
```

With trigger mode armed and **no pulse source connected, zero frames is
correct behavior** — the sensor is waiting. That's also the easiest
smoke test: armed = stream produces nothing; disarmed = free-run returns
(validated on orion 2026-08-01: 0 frames in 8 s armed, 60.38 fps after
clearing).

### 3b. Configuring the device tree (cameras + PWM pin)

`scripts/configure_camera_dt.py` (run ON the Jetson, modeled on
`/opt/nvidia/jetson-io/jetson-io.py`) sets up the boot device tree for the
camera layout and, optionally, a 40-pin-header PWM pin for the trigger
line:

```bash
sudo ./configure_camera_dt.py --cams dual --pwm 32 --set-default --reboot
sudo ./configure_camera_dt.py --cams a               # single cam, CAM0/J20
./configure_camera_dt.py --list                      # inspect, no root
```

It clones the current default boot entry into a script-owned label
(`imx296io`), swaps in the chosen IMX296 overlay, and — for `--pwm` —
regenerates the jetson-io header overlay via NVIDIA's own
`config-by-function.py`, which is **cumulative** (existing header pins
such as a GPS UART survive). The overlay stack is preflighted with
`fdtoverlay` before anything is written; existing labels are never
edited and `extlinux.conf` is backed up. PWM pins: 15 = pwm1
(pwmchip for `3280000.pwm`), 32 = pwm7 (`32e0000.pwm`), 33 = pwm5
(`32c0000.pwm`). A reboot is required (`--reboot`).

> **PWM rate limit (measured on orion, L4T r36):** the PWM controllers
> are clocked at 408 MHz and the divider tops out at 256×8192, so sysfs
> accepts periods only up to ~5.1 ms (≥ ~195 Hz). **30–90 Hz trigger
> rates need the PWM clock reparented to the 32 kHz source** (small DT
> fragment on the pwm node — pending validation) or an external pulse
> source (e.g. the flight controller). Pin-mux and chip enablement are
> validated end-to-end; only the slow-rate clocking remains open.

### 4. Drive the XTR line

Pulse generation is deliberately **not** in this driver (or the GStreamer
element) — one shared line drives both sensors, so it's platform
infrastructure: a Jetson hardware PWM channel or timer-driven GPIO,
configured before the pipeline starts. A userspace GPIO toggle loop is
not acceptable — its jitter defeats the purpose.

Hardware facts (from Raspberry Pi's official *"External trigger on the
GS camera"* documentation, checked 2026-08-02):

- The trigger goes to the **XTR and GND touchpoints on the back of the
  camera PCB** — it is NOT carried on the CSI/FFC ribbon. Solder a fine
  wire to each; the ribbon and everything Jetson-side stay unchanged.
- **XTR is a 1.8 V input.** The official recipe for a 3.3 V driver
  (their example is a Pi Pico on GPIO 28) is a potential divider:
  **1.5 kΩ in series, 1.8 kΩ to ground** → 1.8 V at the pad.
- **Active LOW, pulse width = exposure**: "the exposure time is equal to
  the low pulse-width time plus an additional 14.26 µs". Idle level is
  high; pull XTR low for the exposure duration. Pulse frequency =
  framerate ("a PWM frequency of 30 Hz leads to a framerate of 30 fps").
- **Board mod check**: if the camera board has transistor **Q2** fitted,
  **remove R11** — required for external trigger operation (per the
  official docs; check both boards).
- Don't exceed the mode's free-run maximum rate (~60 fps full frame,
  ~90 fps in the 720p crop — readout still takes a full readout time per
  frame).
- **Not yet validated with real pulses on this rig** — the no-pulse and
  register paths are hardware-validated; the first wired pulse test
  should sanity-check exposure-vs-low-width scaling.

### 5. Timestamping interaction

The Tegra VI stamps start-of-frame in hardware regardless of *why* a
frame arrived, so triggered frames are timestamped exactly like free-run
frames. One consequence of pulse-width exposure: the sensor's `exposure`
V4L2 control does nothing while triggered, so anything doing
mid-exposure correction must be told the pulse width — the companion
`nvimx296camerasrc` element uses its `exposure` property for that (see
`doc/timestamping-and-trigger-howto.md` in that repo).

## Gotchas

- **Video-node numbering swaps between module loads/boots.** Identify
  cameras by bus address, not node number:
  `v4l2-ctl -d /dev/video0 --info | grep bus` (e.g. `9-001a` vs
  `10-001a`).
- While armed with a stopped pulse train, the VI's own capture timeout
  (~2.5 s) fires and recycles empty buffers with the error flag set —
  kernel log noise about capture timeouts in this state is expected, not
  a fault.
- `frame_rate`/`exposure` controls still accept writes in trigger mode;
  they're just inert at the sensor until free-run resumes.

## Implementation map (for maintainers)

| piece | where |
|---|---|
| register defines (`CTRL0B` 0x300b, `LOWLAGTRG` 0x30ae) | `nv_imx296.c` §2 |
| control definition (`TEGRA_CAMERA_CID_BASE+200`) | `imx296_trigger_mode_ctrl_cfg` |
| control registration — in the subdev `.registered` op, **not** probe | `imx296_subdev_registered()` |
| register writes at stream start | `imx296_start_streaming()` |

The `.registered` placement is load-bearing: the VI channel snapshots
subdev controls into `/dev/videoN` synchronously inside
`v4l2_async_register_subdev()` (which `tegracam_v4l2subdev_register()`
calls last), so a control added after that call in probe never reaches
the video node in the normal boot order. Reference implementation for the
sensor registers: RPi kernel `drivers/media/i2c/imx296.c`, trigger_mode=1
path.
