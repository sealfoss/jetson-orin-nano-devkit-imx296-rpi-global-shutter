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

### 4. Drive the XTR line

Pulse generation is deliberately **not** in this driver (or the GStreamer
element) — one shared line drives both sensors, so it's platform
infrastructure: a Jetson hardware PWM channel or timer-driven GPIO,
configured before the pipeline starts. A userspace GPIO toggle loop is
not acceptable — its jitter defeats the purpose.

Hardware notes:

- The XTR pad is exposed on the RPi GS camera board. Consult Raspberry
  Pi's *"External trigger on the GS camera"* documentation for the pad
  location, signal polarity, and level requirements **before wiring** —
  the sensor side is 1.8 V logic territory; do not assume 3.3 V tolerance.
- Pulse width = exposure time. The pulse train period sets the frame
  rate; don't exceed the mode's free-run maximum (~60 fps full frame,
  ~90 fps in the 720p crop — the readout still takes a full readout time
  per frame).
- **Not yet validated with real pulses** — the no-pulse and register
  paths are hardware-validated, but the first wired pulse test should
  confirm polarity and minimum/maximum pulse widths against the RPi docs.

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
