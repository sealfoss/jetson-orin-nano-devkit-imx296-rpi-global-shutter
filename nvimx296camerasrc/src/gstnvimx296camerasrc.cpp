/* SPDX-License-Identifier: GPL-2.0-only
 * nvimx296camerasrc: GStreamer source element for the RPi Global Shutter
 * Camera (Sony IMX296) on Jetson Orin, bypassing Argus entirely:
 *
 *   V4L2 RG10 capture (DMABUF, zero-copy)  -->  fused CUDA ISP  -->  NV12
 *   in memory:NVMM buffers (NvBufSurface), pushed downstream.
 *
 * Zero-copy capture: the VI DMAs each raw frame directly into an
 * NvBufSurface dmabuf that is persistently EGL/CUDA-mapped; the surface
 * pitch is imposed on the VI via the driver's preferred_stride control,
 * so pitches match by construction (no remap_surf-style copies). If the
 * driver refuses the negotiation the element falls back automatically to
 * MMAP + pinned HtoD (zero-copy=false forces the fallback for A/B).
 *
 * Buffer conventions follow NVIDIA's own gst-nvv4l2camera/gst-nvarguscamera
 * (GstMemory of sizeof(NvBufSurface); mapped data IS the NvBufSurface*).
 * All tone/color controls pre-bake into LUT/matrix on property change; the
 * GPU cost is constant. See imx296_isp.cu for the kernel.
 *
 * Deliberate scope notes:
 *  - exposure/gain applied via V4L2 after streaming starts (mode-table
 *    default clobber + no-change-skip findings from bring-up), with the
 *    nudge trick.
 */
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>

#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "imx296_isp.h"

GST_DEBUG_CATEGORY_STATIC(nvimx296_debug);
#define GST_CAT_DEFAULT nvimx296_debug

#define N_CAP_BUFS 4   /* V4L2 MMAP capture buffers        */
#define N_OUT_BUFS 6   /* NVMM output surfaces in the pool */

/* ------------------------------------------------------------------ */
typedef struct {
    NvBufSurface *surf;
    CUgraphicsResource cures;
    uint8_t *dY, *dUV;
    size_t pitchY, pitchUV;
    int idx;
} OutBuf;

/* zero-copy capture buffer: the VI DMAs the RG10 frame straight into an
 * NvBufSurface dmabuf that is persistently EGL/CUDA-mapped. The surface is
 * a byte container (GRAY8, width*2 x height); its color-format tag is
 * cosmetic - only pitch/size matter, and the driver's preferred_stride
 * control is used to make the VI's stride EQUAL the surface pitch. */
typedef struct {
    NvBufSurface *surf;
    CUgraphicsResource cures;
    uint16_t *dptr;        /* device pointer to raw frame */
    int fd;                /* dmabuf fd queued to V4L2 */
} CapBuf;

typedef struct _GstNvImx296CameraSrc {
    GstPushSrc parent;

    /* properties */
    gchar *device;
    gchar *tuning_file;
    gint exposure_us;
    gint gain;             /* dB*10 */
    IspParams isp_params;
    gboolean params_dirty, ctrl_dirty;

    /* negotiated mode */
    gint width, height, fps_n, fps_d;

    /* v4l2 */
    int fd;
    struct { void *start; size_t length; } capbuf[N_CAP_BUFS];  /* MMAP fallback */
    CapBuf cap[N_CAP_BUFS];                                     /* zero-copy */
    gboolean zerocopy_active;
    gboolean prop_zerocopy;
    guint32 bytesperline;
    unsigned ctrl_id_exposure, ctrl_id_gain, ctrl_id_frame_rate, ctrl_id_stride;
    gboolean streaming;

    /* cuda / egl */
    EGLDisplay egl_dpy;
    CUcontext cuctx;
    cudaStream_t stream;
    uint16_t *d_raw;       /* staging device buffer */
    size_t d_raw_pitch;
    IspCore *core;
    IspTuning tuning;

    /* output pool */
    OutBuf out[N_OUT_BUFS];
    GAsyncQueue *freeq;    /* holds OutBuf* */

    guint64 frame_no;
    GMutex lock;
} GstNvImx296CameraSrc;

typedef struct _GstNvImx296CameraSrcClass {
    GstPushSrcClass parent_class;
} GstNvImx296CameraSrcClass;

G_DEFINE_TYPE(GstNvImx296CameraSrc, gst_nvimx296camerasrc, GST_TYPE_PUSH_SRC)
#define NVIMX296(obj) ((GstNvImx296CameraSrc *)(obj))

/* ------------------------------------------------------------------ */
enum {
    PROP_0, PROP_DEVICE, PROP_TUNING_FILE,
    PROP_EXPOSURE, PROP_GAIN,
    PROP_TONE_PRESET, PROP_CONTRAST, PROP_BRIGHTNESS, PROP_SATURATION,
    PROP_DIGITAL_GAIN, PROP_DITHER, PROP_BLACK_OFFSET,
    PROP_KNEE_POINT, PROP_KNEE_STRENGTH, PROP_TONE_LUT_FILE,
    PROP_AWB_MODE, PROP_AWB_CT, PROP_FLIP180, PROP_ZEROCOPY,
};

#define TONE_PRESET_TYPE (tone_preset_get_type())
static GType tone_preset_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { TONE_TUNING, "RPi tuning-file curve", "tuning" },
        { TONE_SRGB,   "sRGB gamma",            "srgb" },
        { TONE_REC709, "Rec.709 gamma",         "rec709" },
        { TONE_LINEAR, "Linear passthrough",    "linear" },
        { 0, NULL, NULL },
    };
    if (!t) t = g_enum_register_static("NvImx296TonePreset", v);
    return t;
}
#define AWB_MODE_TYPE (awb_mode_get_type())
static GType awb_mode_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { AWB_AUTO,   "Grey-world constrained to tuning CT curve", "auto" },
        { AWB_TUNING, "Fixed gains from tuning curve at awb-ct",   "tuning" },
        { AWB_OFF,    "Unity gains",                               "off" },
        { 0, NULL, NULL },
    };
    if (!t) t = g_enum_register_static("NvImx296AwbMode", v);
    return t;
}

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw(memory:NVMM), format=NV12, "
        "width=1456, height=1088, framerate=[1/1,60/1]; "
        "video/x-raw(memory:NVMM), format=NV12, "
        "width=1280, height=720, framerate=[1/1,90/1]"));

/* ================== V4L2 helpers ================== */

/* Match by v4l2-ctl's normalized naming (lowercase, spaces->underscores):
 * the driver's real control names are "Exposure", "Frame Rate", ... while
 * everything user-facing in this project uses the normalized form. */
static unsigned v4l2_find_ctrl(int fd, const char *name)
{
    struct v4l2_query_ext_ctrl q;
    memset(&q, 0, sizeof(q));
    q.id = V4L2_CTRL_FLAG_NEXT_CTRL | V4L2_CTRL_FLAG_NEXT_COMPOUND;
    while (ioctl(fd, VIDIOC_QUERY_EXT_CTRL, &q) == 0) {
        char norm[64];
        size_t i;
        for (i = 0; i < sizeof(norm) - 1 && q.name[i]; i++)
            norm[i] = (q.name[i] == ' ') ? '_' : g_ascii_tolower(q.name[i]);
        norm[i] = 0;
        if (!g_strcmp0(norm, name)) return q.id;
        q.id |= V4L2_CTRL_FLAG_NEXT_CTRL | V4L2_CTRL_FLAG_NEXT_COMPOUND;
    }
    return 0;
}

static int v4l2_set_ctrl64(int fd, unsigned id, gint64 val)
{
    struct v4l2_ext_control c;
    struct v4l2_ext_controls cs;
    memset(&c, 0, sizeof(c)); memset(&cs, 0, sizeof(cs));
    c.id = id; c.value64 = val;
    cs.which = V4L2_CTRL_WHICH_CUR_VAL; cs.count = 1; cs.controls = &c;
    return ioctl(fd, VIDIOC_S_EXT_CTRLS, &cs);
}

/* set with the no-change-skip defeating nudge (bring-up finding) */
static void v4l2_set_ctrl64_nudge(int fd, unsigned id, gint64 val, gint64 lo)
{
    if (!id) return;
    gint64 nudge = (val - 1 >= lo) ? val - 1 : val + 1;
    v4l2_set_ctrl64(fd, id, nudge);
    v4l2_set_ctrl64(fd, id, val);
}

static gboolean v4l2_setup(GstNvImx296CameraSrc *self)
{
    self->fd = open(self->device, O_RDWR | O_NONBLOCK);
    if (self->fd < 0) {
        GST_ERROR_OBJECT(self, "cannot open %s", self->device);
        return FALSE;
    }

    self->ctrl_id_exposure   = v4l2_find_ctrl(self->fd, "exposure");
    self->ctrl_id_gain       = v4l2_find_ctrl(self->fd, "gain");
    self->ctrl_id_frame_rate = v4l2_find_ctrl(self->fd, "frame_rate");
    self->ctrl_id_stride     = v4l2_find_ctrl(self->fd, "preferred_stride");

    /* 64-byte-aligned stride: THE Orin VI empiric. Set before S_FMT. */
    guint32 want_stride = ((self->width * 2) + 63) / 64 * 64;
    if (self->ctrl_id_stride)
        v4l2_set_ctrl64(self->fd, self->ctrl_id_stride, want_stride);

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = self->width;
    fmt.fmt.pix.height = self->height;
    fmt.fmt.pix.pixelformat = v4l2_fourcc('R', 'G', '1', '0');
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(self->fd, VIDIOC_S_FMT, &fmt) < 0) {
        GST_ERROR_OBJECT(self, "S_FMT failed: %s", g_strerror(errno));
        return FALSE;
    }
    if (self->ctrl_id_stride)  /* driver may only latch it post-S_FMT */
        v4l2_set_ctrl64(self->fd, self->ctrl_id_stride, want_stride);
    /* re-read: bytesperline is what the VI will actually write */
    if (ioctl(self->fd, VIDIOC_G_FMT, &fmt) < 0) return FALSE;
    self->bytesperline = fmt.fmt.pix.bytesperline;
    if (self->bytesperline < (guint32)self->width * 2 || (self->bytesperline & 63)) {
        GST_ERROR_OBJECT(self, "unusable bytesperline %u (need 64-aligned >= %d)",
                         self->bytesperline, self->width * 2);
        return FALSE;
    }
    GST_INFO_OBJECT(self, "V4L2 %dx%d RG10, bytesperline=%u",
                    self->width, self->height, self->bytesperline);

    return TRUE;   /* buffers + STREAMON happen in v4l2_buffers_setup() */
}

/* Allocate one GRAY8 byte-container surface of the raw frame geometry and
 * EGL/CUDA-map it. Returns FALSE on any failure (caller cleans up). */
static gboolean capbuf_alloc(GstNvImx296CameraSrc *self, CapBuf *cb)
{
    NvBufSurfaceAllocateParams p;
    memset(&p, 0, sizeof(p));
    p.params.width = self->width * 2;      /* bytes per row of RG10 */
    p.params.height = self->height;
    p.params.layout = NVBUF_LAYOUT_PITCH;
    p.params.memType = NVBUF_MEM_DEFAULT;
    p.params.gpuId = 0;
    p.params.colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
    if (NvBufSurfaceAllocate(&cb->surf, 1, &p) != 0) return FALSE;
    cb->surf->numFilled = 1;
    cb->fd = (int)cb->surf->surfaceList[0].bufferDesc;

    if (NvBufSurfaceMapEglImage(cb->surf, 0) != 0) return FALSE;
    if (cuGraphicsEGLRegisterImage(&cb->cures,
            cb->surf->surfaceList[0].mappedAddr.eglImage,
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) return FALSE;
    CUeglFrame f;
    if (cuGraphicsResourceGetMappedEglFrame(&f, cb->cures, 0, 0) != CUDA_SUCCESS)
        return FALSE;
    cb->dptr = (uint16_t *)f.frame.pPitch[0];
    return TRUE;
}

static void capbufs_teardown(GstNvImx296CameraSrc *self)
{
    for (int i = 0; i < N_CAP_BUFS; i++) {
        CapBuf *cb = &self->cap[i];
        if (cb->cures) { cuGraphicsUnregisterResource(cb->cures); cb->cures = NULL; }
        if (cb->surf) {
            NvBufSurfaceUnMapEglImage(cb->surf, 0);
            NvBufSurfaceDestroy(cb->surf);
            cb->surf = NULL;
        }
        cb->dptr = NULL; cb->fd = -1;
    }
}

/* Zero-copy path: allocate capture surfaces, force the VI's stride to the
 * surface pitch via preferred_stride, and queue the dmabufs to V4L2.
 * Requires CUDA/EGL to be initialized. Returns FALSE -> caller falls back
 * to MMAP+HtoD. */
static gboolean v4l2_buffers_setup_dmabuf(GstNvImx296CameraSrc *self)
{
    if (!self->prop_zerocopy || !self->ctrl_id_stride) return FALSE;

    /* probe surface: what pitch does the allocator want for this width? */
    if (!capbuf_alloc(self, &self->cap[0])) { capbufs_teardown(self); return FALSE; }
    guint32 pitch = self->cap[0].surf->surfaceList[0].planeParams.pitch[0];
    if (pitch & 63) { capbufs_teardown(self); return FALSE; }  /* VI needs 64-aligned */

    /* re-negotiate the VI stride to exactly the surface pitch */
    v4l2_set_ctrl64(self->fd, self->ctrl_id_stride, pitch);
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = self->width;
    fmt.fmt.pix.height = self->height;
    fmt.fmt.pix.pixelformat = v4l2_fourcc('R', 'G', '1', '0');
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(self->fd, VIDIOC_S_FMT, &fmt) < 0) { capbufs_teardown(self); return FALSE; }
    v4l2_set_ctrl64(self->fd, self->ctrl_id_stride, pitch);
    if (ioctl(self->fd, VIDIOC_G_FMT, &fmt) < 0) { capbufs_teardown(self); return FALSE; }
    if (fmt.fmt.pix.bytesperline != pitch) {
        GST_WARNING_OBJECT(self,
            "zero-copy: driver bytesperline %u != surface pitch %u - falling back",
            fmt.fmt.pix.bytesperline, pitch);
        capbufs_teardown(self);
        return FALSE;
    }
    self->bytesperline = pitch;

    for (int i = 1; i < N_CAP_BUFS; i++)
        if (!capbuf_alloc(self, &self->cap[i])) { capbufs_teardown(self); return FALSE; }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = N_CAP_BUFS; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(self->fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        GST_WARNING_OBJECT(self, "zero-copy: REQBUFS(DMABUF) failed - falling back");
        capbufs_teardown(self);
        return FALSE;
    }
    for (int i = 0; i < N_CAP_BUFS; i++) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_DMABUF;
        b.index = i; b.m.fd = self->cap[i].fd;
        if (ioctl(self->fd, VIDIOC_QBUF, &b) < 0) {
            GST_WARNING_OBJECT(self, "zero-copy: QBUF(DMABUF) failed: %s - falling back",
                               g_strerror(errno));
            struct v4l2_requestbuffers rel;
            memset(&rel, 0, sizeof(rel));
            rel.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rel.memory = V4L2_MEMORY_DMABUF;
            ioctl(self->fd, VIDIOC_REQBUFS, &rel);
            capbufs_teardown(self);
            return FALSE;
        }
    }
    GST_INFO_OBJECT(self, "zero-copy capture active: %d dmabufs, stride %u",
                    N_CAP_BUFS, pitch);
    return TRUE;
}

/* MMAP + pinned-HtoD fallback (the proven v1 path). */
static gboolean v4l2_buffers_setup_mmap(GstNvImx296CameraSrc *self)
{
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = N_CAP_BUFS; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(self->fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        GST_ERROR_OBJECT(self, "REQBUFS failed");
        return FALSE;
    }
    for (unsigned i = 0; i < req.count && i < N_CAP_BUFS; i++) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
        if (ioctl(self->fd, VIDIOC_QUERYBUF, &b) < 0) return FALSE;
        self->capbuf[i].length = b.length;
        self->capbuf[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, self->fd, b.m.offset);
        if (self->capbuf[i].start == MAP_FAILED) return FALSE;
        /* pin for fast async HtoD */
        cudaHostRegister(self->capbuf[i].start, b.length, cudaHostRegisterDefault);
        if (ioctl(self->fd, VIDIOC_QBUF, &b) < 0) return FALSE;
    }
    return TRUE;
}

static gboolean v4l2_buffers_setup(GstNvImx296CameraSrc *self)
{
    self->zerocopy_active = v4l2_buffers_setup_dmabuf(self);
    if (!self->zerocopy_active) {
        if (!v4l2_buffers_setup_mmap(self)) return FALSE;
        /* MMAP path needs the device staging buffer */
        if (cudaMallocPitch((void **)&self->d_raw, &self->d_raw_pitch,
                            self->bytesperline, self->height) != cudaSuccess)
            return FALSE;
    }

    enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(self->fd, VIDIOC_STREAMON, &t) < 0) {
        GST_ERROR_OBJECT(self, "STREAMON failed: %s", g_strerror(errno));
        return FALSE;
    }
    self->streaming = TRUE;
    return TRUE;
}

static void v4l2_apply_controls(GstNvImx296CameraSrc *self)
{
    v4l2_set_ctrl64_nudge(self->fd, self->ctrl_id_exposure, self->exposure_us, 15);
    v4l2_set_ctrl64_nudge(self->fd, self->ctrl_id_gain, self->gain, 0);
    gint64 fr = (gint64)self->fps_n * 1000000 / (self->fps_d ? self->fps_d : 1);
    v4l2_set_ctrl64_nudge(self->fd, self->ctrl_id_frame_rate, fr, 1000000);
}

static void capbufs_teardown(GstNvImx296CameraSrc *self);  /* fwd */

static void v4l2_teardown(GstNvImx296CameraSrc *self)
{
    if (self->fd >= 0) {
        if (self->streaming) {
            enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(self->fd, VIDIOC_STREAMOFF, &t);
            self->streaming = FALSE;
        }
        for (int i = 0; i < N_CAP_BUFS; i++) {
            if (self->capbuf[i].start) {
                cudaHostUnregister(self->capbuf[i].start);
                munmap(self->capbuf[i].start, self->capbuf[i].length);
                self->capbuf[i].start = NULL;
            }
        }
        close(self->fd);
        self->fd = -1;
    }
    /* after STREAMOFF + close the VI holds no references to the dmabufs */
    capbufs_teardown(self);
    self->zerocopy_active = FALSE;
}

/* ================== EGL / CUDA / pool ================== */

static gboolean egl_cuda_init_once(GstNvImx296CameraSrc *self)
{
    /* headless-capable display: EGL device platform first */
    PFNEGLQUERYDEVICESEXTPROC qdev =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    self->egl_dpy = EGL_NO_DISPLAY;
    if (qdev && gpd) {
        EGLDeviceEXT devs[4]; EGLint n = 0;
        if (qdev(4, devs, &n) && n > 0)
            self->egl_dpy = gpd(EGL_PLATFORM_DEVICE_EXT, devs[0], NULL);
    }
    if (self->egl_dpy == EGL_NO_DISPLAY)
        self->egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (self->egl_dpy == EGL_NO_DISPLAY || !eglInitialize(self->egl_dpy, NULL, NULL)) {
        GST_ERROR_OBJECT(self, "EGL display init failed");
        return FALSE;
    }
    if (cuInit(0) != CUDA_SUCCESS) return FALSE;
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return FALSE;
    if (cuDevicePrimaryCtxRetain(&self->cuctx, dev) != CUDA_SUCCESS) return FALSE;
    cuCtxSetCurrent(self->cuctx);
    if (cudaStreamCreate(&self->stream) != cudaSuccess) return FALSE;
    return TRUE;
}

static gboolean pool_setup(GstNvImx296CameraSrc *self)
{
    self->freeq = g_async_queue_new();
    for (int i = 0; i < N_OUT_BUFS; i++) {
        OutBuf *ob = &self->out[i];
        ob->idx = i;

        NvBufSurfaceAllocateParams p;
        memset(&p, 0, sizeof(p));
        p.params.width = self->width;
        p.params.height = self->height;
        p.params.layout = NVBUF_LAYOUT_PITCH;
        p.params.memType = NVBUF_MEM_DEFAULT;
        p.params.gpuId = 0;
        p.params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        if (NvBufSurfaceAllocate(&ob->surf, 1, &p) != 0) {
            GST_ERROR_OBJECT(self, "NvBufSurfaceAllocate failed");
            return FALSE;
        }
        ob->surf->numFilled = 1;

        if (NvBufSurfaceMapEglImage(ob->surf, 0) != 0) {
            GST_ERROR_OBJECT(self, "MapEglImage failed");
            return FALSE;
        }
        if (cuGraphicsEGLRegisterImage(&ob->cures,
                ob->surf->surfaceList[0].mappedAddr.eglImage,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
            GST_ERROR_OBJECT(self, "cuGraphicsEGLRegisterImage failed");
            return FALSE;
        }
        CUeglFrame f;
        if (cuGraphicsResourceGetMappedEglFrame(&f, ob->cures, 0, 0) != CUDA_SUCCESS)
            return FALSE;
        if (f.frameType != CU_EGL_FRAME_TYPE_PITCH || f.planeCount < 2) {
            GST_ERROR_OBJECT(self, "unexpected EGL frame layout (type %d planes %d)",
                             f.frameType, f.planeCount);
            return FALSE;
        }
        ob->dY  = (uint8_t *)f.frame.pPitch[0];
        ob->dUV = (uint8_t *)f.frame.pPitch[1];
        ob->pitchY  = ob->surf->surfaceList[0].planeParams.pitch[0];
        ob->pitchUV = ob->surf->surfaceList[0].planeParams.pitch[1];

        g_async_queue_push(self->freeq, ob);
    }
    return TRUE;
}

static void pool_teardown(GstNvImx296CameraSrc *self)
{
    if (self->freeq) { g_async_queue_unref(self->freeq); self->freeq = NULL; }
    for (int i = 0; i < N_OUT_BUFS; i++) {
        OutBuf *ob = &self->out[i];
        if (ob->cures) { cuGraphicsUnregisterResource(ob->cures); ob->cures = NULL; }
        if (ob->surf) {
            NvBufSurfaceUnMapEglImage(ob->surf, 0);
            NvBufSurfaceDestroy(ob->surf);
            ob->surf = NULL;
        }
    }
}

/* downstream released the wrapped buffer: back to the free list */
typedef struct { GstNvImx296CameraSrc *self; OutBuf *ob; } RecycleCtx;
static void recycle_notify(gpointer data)
{
    RecycleCtx *ctx = (RecycleCtx *)data;
    /* element may be shutting down: freeq presence gates the push */
    g_mutex_lock(&ctx->self->lock);
    if (ctx->self->freeq)
        g_async_queue_push(ctx->self->freeq, ctx->ob);
    g_mutex_unlock(&ctx->self->lock);
    gst_object_unref(ctx->self);
    g_free(ctx);
}

/* ================== GstBaseSrc virtuals ================== */

/* Full capture/GPU configuration happens HERE, not in start():
 * GstBaseSrc calls start() before caps are negotiated, and every piece of
 * this element (V4L2 mode, staging buffer, NVMM pool, ISP core) depends on
 * the negotiated geometry. */
static void teardown_pipeline_state(GstNvImx296CameraSrc *self)
{
    v4l2_teardown(self);
    if (self->core) { isp_core_destroy(self->core); self->core = NULL; }
    g_mutex_lock(&self->lock);
    GAsyncQueue *q = self->freeq;
    self->freeq = NULL;
    g_mutex_unlock(&self->lock);
    if (q) g_async_queue_unref(q);
    /* NOTE: assumes no downstream element still holds our buffers at
     * reconfigure time (true for fixed-caps pipelines; renegotiation
     * mid-stream with in-flight buffers is out of v1 scope). */
    pool_teardown(self);
    if (self->d_raw) { cudaFree(self->d_raw); self->d_raw = NULL; }
}

static gboolean nvimx296_set_caps(GstBaseSrc *bsrc, GstCaps *caps)
{
    GstNvImx296CameraSrc *self = NVIMX296(bsrc);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) return FALSE;

    gint w = GST_VIDEO_INFO_WIDTH(&info), h = GST_VIDEO_INFO_HEIGHT(&info);
    gint fn = GST_VIDEO_INFO_FPS_N(&info), fd = GST_VIDEO_INFO_FPS_D(&info);
    GST_INFO_OBJECT(self, "caps: %dx%d @ %d/%d", w, h, fn, fd);

    if (self->core && w == self->width && h == self->height) {
        /* geometry unchanged: only the frame rate may need updating */
        self->fps_n = fn; self->fps_d = fd;
        self->ctrl_dirty = TRUE;
        return TRUE;
    }

    teardown_pipeline_state(self);
    self->width = w; self->height = h; self->fps_n = fn; self->fps_d = fd;

    if (!v4l2_setup(self)) return FALSE;
    if (!self->stream) {          /* one-time CUDA/EGL init */
        if (!egl_cuda_init_once(self)) return FALSE;
    }
    cuCtxSetCurrent(self->cuctx);
    /* buffer setup needs CUDA/EGL live (zero-copy capture surfaces) */
    if (!v4l2_buffers_setup(self)) return FALSE;
    if (!pool_setup(self)) return FALSE;

    self->core = isp_core_create(&self->tuning, &self->isp_params,
                                 self->width, self->height, self->stream);
    if (!self->core) return FALSE;

    v4l2_apply_controls(self);
    self->ctrl_dirty = TRUE;   /* re-apply once frames flow (proven ordering) */
    self->frame_no = 0;
    return TRUE;
}

static GstCaps *nvimx296_fixate(GstBaseSrc *bsrc, GstCaps *caps)
{
    caps = gst_caps_make_writable(caps);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gint w = 1456;
    gst_structure_get_int(s, "width", &w);
    gst_structure_fixate_field_nearest_int(s, "width", 1456);
    gst_structure_get_int(s, "width", &w);
    gst_structure_fixate_field_nearest_fraction(s, "framerate",
                                                w == 1280 ? 90 : 60, 1);
    caps = GST_BASE_SRC_CLASS(gst_nvimx296camerasrc_parent_class)->fixate(bsrc, caps);
    return caps;
}

static gboolean nvimx296_start(GstBaseSrc *bsrc)
{
    GstNvImx296CameraSrc *self = NVIMX296(bsrc);
    /* geometry-dependent setup is deferred to set_caps(); here only the
     * tuning data, which is mode-independent. */
    const char *tf = self->tuning_file;
    if (isp_tuning_load(tf, &self->tuning) != 0) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
                          ("tuning file %s unusable", tf), (NULL));
        return FALSE;
    }
    self->isp_params.awb_ct = self->isp_params.awb_ct ? self->isp_params.awb_ct : 4560.0;
    return TRUE;
}

static gboolean nvimx296_stop(GstBaseSrc *bsrc)
{
    GstNvImx296CameraSrc *self = NVIMX296(bsrc);
    g_mutex_lock(&self->lock);
    GAsyncQueue *q = self->freeq;
    self->freeq = NULL;
    g_mutex_unlock(&self->lock);
    teardown_pipeline_state(self);
    if (q) g_async_queue_unref(q);
    if (self->stream) { cudaStreamDestroy(self->stream); self->stream = NULL; }
    self->width = 0;  /* force full reconfigure on next set_caps */
    return TRUE;
}

static GstFlowReturn nvimx296_create(GstPushSrc *psrc, GstBuffer **outbuf)
{
    GstNvImx296CameraSrc *self = NVIMX296(psrc);

    /* 1. free output surface (waits for downstream to release one) */
    g_mutex_lock(&self->lock);
    GAsyncQueue *q = self->freeq ? g_async_queue_ref(self->freeq) : NULL;
    g_mutex_unlock(&self->lock);
    if (!q) return GST_FLOW_FLUSHING;
    OutBuf *ob = (OutBuf *)g_async_queue_timeout_pop(q, 2 * G_TIME_SPAN_SECOND);
    g_async_queue_unref(q);
    if (!ob) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("no free output buffer in 2s (downstream stuck?)"), (NULL));
        return GST_FLOW_ERROR;
    }

    /* 2. DQBUF with poll timeout (VI-wedge turns into an error, not a hang) */
    struct pollfd pfd = { self->fd, POLLIN, 0 };
    int pr = poll(&pfd, 1, 2000);
    if (pr <= 0) {
        GST_ELEMENT_ERROR(self, RESOURCE, READ,
                          ("no frame from %s within 2s (VI stalled?)", self->device), (NULL));
        return GST_FLOW_ERROR;
    }
    struct v4l2_buffer b;
    memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = self->zerocopy_active ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
    if (ioctl(self->fd, VIDIOC_DQBUF, &b) < 0) {
        GST_ELEMENT_ERROR(self, RESOURCE, READ, ("DQBUF: %s", g_strerror(errno)), (NULL));
        return GST_FLOW_ERROR;
    }
    if (self->zerocopy_active) /* QBUF wants the fd repopulated */
        b.m.fd = self->cap[b.index].fd;

    /* 3. late control (re)apply: first frames + property changes */
    if (self->ctrl_dirty) {
        v4l2_apply_controls(self);
        self->ctrl_dirty = FALSE;
    }
    if (self->params_dirty) {
        g_mutex_lock(&self->lock);
        IspParams p = self->isp_params;
        self->params_dirty = FALSE;
        g_mutex_unlock(&self->lock);
        isp_core_update_params(self->core, &p);
    }

    /* 4. fused kernel + sync, then requeue the capture buffer.
     * zero-copy: the kernel reads the VI's dmabuf in place (pitch ==
     * bytesperline by construction). MMAP fallback: pinned HtoD first. */
    cuCtxSetCurrent(self->cuctx);
    if (self->zerocopy_active) {
        isp_core_process_nv12(self->core, self->cap[b.index].dptr,
                              self->bytesperline,
                              ob->dY, ob->pitchY, ob->dUV, ob->pitchUV,
                              (unsigned)self->frame_no);
    } else {
        cudaMemcpy2DAsync(self->d_raw, self->d_raw_pitch,
                          self->capbuf[b.index].start, self->bytesperline,
                          self->width * 2, self->height,
                          cudaMemcpyHostToDevice, self->stream);
        isp_core_process_nv12(self->core, self->d_raw, self->d_raw_pitch,
                              ob->dY, ob->pitchY, ob->dUV, ob->pitchUV,
                              (unsigned)self->frame_no);
    }
    cudaError_t ce = cudaStreamSynchronize(self->stream);
    ioctl(self->fd, VIDIOC_QBUF, &b);
    if (ce != cudaSuccess) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("CUDA: %s", cudaGetErrorString(ce)), (NULL));
        return GST_FLOW_ERROR;
    }
    if (isp_core_awb_tick(self->core))
        GST_LOG_OBJECT(self, "AWB updated");

    /* 5. wrap the NvBufSurface as the downstream NVMM convention expects */
    RecycleCtx *ctx = g_new0(RecycleCtx, 1);
    ctx->self = (GstNvImx296CameraSrc *)gst_object_ref(self);
    ctx->ob = ob;
    GstBuffer *buf = gst_buffer_new_wrapped_full(
        (GstMemoryFlags)GST_MEMORY_FLAG_READONLY,
        ob->surf, sizeof(NvBufSurface), 0, sizeof(NvBufSurface),
        ctx, recycle_notify);

    if (self->fps_n > 0)
        GST_BUFFER_DURATION(buf) =
            gst_util_uint64_scale_int(GST_SECOND, self->fps_d, self->fps_n);
    self->frame_no++;
    *outbuf = buf;
    return GST_FLOW_OK;
}

/* ================== properties ================== */

static void nvimx296_set_property(GObject *obj, guint prop_id,
                                  const GValue *value, GParamSpec *pspec)
{
    GstNvImx296CameraSrc *self = NVIMX296(obj);
    g_mutex_lock(&self->lock);
    switch (prop_id) {
    case PROP_DEVICE: g_free(self->device); self->device = g_value_dup_string(value); break;
    case PROP_TUNING_FILE: g_free(self->tuning_file); self->tuning_file = g_value_dup_string(value); break;
    case PROP_EXPOSURE: self->exposure_us = g_value_get_int(value); self->ctrl_dirty = TRUE; break;
    case PROP_GAIN: self->gain = g_value_get_int(value); self->ctrl_dirty = TRUE; break;
    case PROP_TONE_PRESET: self->isp_params.preset = (TonePreset)g_value_get_enum(value); self->params_dirty = TRUE; break;
    case PROP_CONTRAST: self->isp_params.contrast = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_BRIGHTNESS: self->isp_params.brightness = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_SATURATION: self->isp_params.saturation = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_DIGITAL_GAIN: self->isp_params.digital_gain = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_DITHER: self->isp_params.dither = g_value_get_boolean(value); self->params_dirty = TRUE; break;
    case PROP_BLACK_OFFSET: self->isp_params.black_offset = g_value_get_int(value); self->params_dirty = TRUE; break;
    case PROP_KNEE_POINT: self->isp_params.knee_point = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_KNEE_STRENGTH: self->isp_params.knee_strength = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_TONE_LUT_FILE: {
        const gchar *s = g_value_get_string(value);
        g_strlcpy(self->isp_params.lut_file, s ? s : "", sizeof(self->isp_params.lut_file));
        self->params_dirty = TRUE; break;
    }
    case PROP_AWB_MODE: self->isp_params.awb = (AwbMode)g_value_get_enum(value); self->params_dirty = TRUE; break;
    case PROP_AWB_CT: self->isp_params.awb_ct = g_value_get_double(value); self->params_dirty = TRUE; break;
    case PROP_FLIP180: self->isp_params.flip180 = g_value_get_boolean(value); self->params_dirty = TRUE; break;
    case PROP_ZEROCOPY: self->prop_zerocopy = g_value_get_boolean(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec); break;
    }
    g_mutex_unlock(&self->lock);
}

static void nvimx296_get_property(GObject *obj, guint prop_id,
                                  GValue *value, GParamSpec *pspec)
{
    GstNvImx296CameraSrc *self = NVIMX296(obj);
    switch (prop_id) {
    case PROP_DEVICE: g_value_set_string(value, self->device); break;
    case PROP_TUNING_FILE: g_value_set_string(value, self->tuning_file); break;
    case PROP_EXPOSURE: g_value_set_int(value, self->exposure_us); break;
    case PROP_GAIN: g_value_set_int(value, self->gain); break;
    case PROP_TONE_PRESET: g_value_set_enum(value, self->isp_params.preset); break;
    case PROP_CONTRAST: g_value_set_double(value, self->isp_params.contrast); break;
    case PROP_BRIGHTNESS: g_value_set_double(value, self->isp_params.brightness); break;
    case PROP_SATURATION: g_value_set_double(value, self->isp_params.saturation); break;
    case PROP_DIGITAL_GAIN: g_value_set_double(value, self->isp_params.digital_gain); break;
    case PROP_DITHER: g_value_set_boolean(value, self->isp_params.dither); break;
    case PROP_BLACK_OFFSET: g_value_set_int(value, self->isp_params.black_offset); break;
    case PROP_KNEE_POINT: g_value_set_double(value, self->isp_params.knee_point); break;
    case PROP_KNEE_STRENGTH: g_value_set_double(value, self->isp_params.knee_strength); break;
    case PROP_TONE_LUT_FILE: g_value_set_string(value, self->isp_params.lut_file); break;
    case PROP_AWB_MODE: g_value_set_enum(value, self->isp_params.awb); break;
    case PROP_AWB_CT: g_value_set_double(value, self->isp_params.awb_ct); break;
    case PROP_FLIP180: g_value_set_boolean(value, self->isp_params.flip180); break;
    case PROP_ZEROCOPY: g_value_set_boolean(value, self->prop_zerocopy); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec); break;
    }
}

static void nvimx296_finalize(GObject *obj)
{
    GstNvImx296CameraSrc *self = NVIMX296(obj);
    g_free(self->device);
    g_free(self->tuning_file);
    g_mutex_clear(&self->lock);
    G_OBJECT_CLASS(gst_nvimx296camerasrc_parent_class)->finalize(obj);
}

/* ================== boilerplate ================== */

static void gst_nvimx296camerasrc_class_init(GstNvImx296CameraSrcClass *klass)
{
    GObjectClass *gobject = G_OBJECT_CLASS(klass);
    GstElementClass *element = GST_ELEMENT_CLASS(klass);
    GstBaseSrcClass *basesrc = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *pushsrc = GST_PUSH_SRC_CLASS(klass);

    gobject->set_property = nvimx296_set_property;
    gobject->get_property = nvimx296_get_property;
    gobject->finalize = nvimx296_finalize;

    basesrc->set_caps = nvimx296_set_caps;
    basesrc->fixate = nvimx296_fixate;
    basesrc->start = nvimx296_start;
    basesrc->stop = nvimx296_stop;
    pushsrc->create = nvimx296_create;

    #define PFLAGS ((GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_PLAYING))
    g_object_class_install_property(gobject, PROP_DEVICE,
        g_param_spec_string("device", "Device", "V4L2 device node", "/dev/video0",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject, PROP_TUNING_FILE,
        g_param_spec_string("tuning-file", "Tuning file",
            "RPi libcamera imx296 tuning JSON",
            "/home/reed/imx296-test/imx296_16mm.json",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject, PROP_EXPOSURE,
        g_param_spec_int("exposure", "Exposure (us)",
            "Sensor exposure in microseconds (8333 = mains-ripple immune)",
            15, 15699, 8333, PFLAGS));
    g_object_class_install_property(gobject, PROP_GAIN,
        g_param_spec_int("gain", "Gain (dB*10)",
            "Sensor analog gain, dB*10 (60 = 6 dB)", 0, 480, 60, PFLAGS));
    g_object_class_install_property(gobject, PROP_TONE_PRESET,
        g_param_spec_enum("tone-preset", "Tone preset", "Base tone curve",
            TONE_PRESET_TYPE, TONE_TUNING, PFLAGS));
    g_object_class_install_property(gobject, PROP_CONTRAST,
        g_param_spec_double("contrast", "Contrast", "S-curve about mid (1=neutral)",
            0.0, 2.0, 1.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_BRIGHTNESS,
        g_param_spec_double("brightness", "Brightness", "Post-curve offset",
            -1.0, 1.0, 0.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_SATURATION,
        g_param_spec_double("saturation", "Saturation", "Chroma scale (1=neutral)",
            0.0, 2.0, 1.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_DIGITAL_GAIN,
        g_param_spec_double("digital-gain", "Digital gain", "Linear pre-curve gain",
            0.25, 4.0, 1.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_DITHER,
        g_param_spec_boolean("dither", "Dither", "Blue-noise dither at 8-bit quantize",
            TRUE, PFLAGS));
    g_object_class_install_property(gobject, PROP_BLACK_OFFSET,
        g_param_spec_int("black-offset", "Black offset",
            "Manual black level trim, 10-bit counts (BLKLEVELAUTO is disabled)",
            -32, 32, 0, PFLAGS));
    g_object_class_install_property(gobject, PROP_KNEE_POINT,
        g_param_spec_double("knee-point", "Knee point",
            "Highlight knee start (linear domain; 1.0 = off)", 0.5, 1.0, 1.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_KNEE_STRENGTH,
        g_param_spec_double("knee-strength", "Knee strength", "Highlight rolloff amount",
            0.0, 1.0, 0.0, PFLAGS));
    g_object_class_install_property(gobject, PROP_TONE_LUT_FILE,
        g_param_spec_string("tone-lut-file", "Tone LUT file",
            "Optional 1024-entry curve override (text, one value per line)",
            "", PFLAGS));
    g_object_class_install_property(gobject, PROP_AWB_MODE,
        g_param_spec_enum("awb", "AWB mode", "White balance mode",
            AWB_MODE_TYPE, AWB_AUTO, PFLAGS));
    g_object_class_install_property(gobject, PROP_ZEROCOPY,
        g_param_spec_boolean("zero-copy", "Zero-copy capture",
            "Capture via V4L2 DMABUF directly into GPU-mapped surfaces "
            "(auto-falls back to MMAP+copy if the driver refuses)",
            TRUE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject, PROP_FLIP180,
        g_param_spec_boolean("flip", "Flip 180",
            "Rotate output 180 degrees (upside-down camera mount)",
            FALSE, PFLAGS));
    g_object_class_install_property(gobject, PROP_AWB_CT,
        g_param_spec_double("awb-ct", "AWB CT",
            "Fixed color temperature for awb=tuning", 2000.0, 10000.0, 4560.0, PFLAGS));

    gst_element_class_add_static_pad_template(element, &src_template);
    gst_element_class_set_static_metadata(element,
        "NVIDIA IMX296 raw camera source with CUDA ISP",
        "Source/Video",
        "RPi Global Shutter Camera (IMX296) V4L2 raw capture with fused CUDA "
        "ISP (RPi tuning data), NV12 memory:NVMM output; bypasses Argus",
        "JetsonPhotonIMX296 project");
}

static void gst_nvimx296camerasrc_init(GstNvImx296CameraSrc *self)
{
    self->device = g_strdup("/dev/video0");
    self->tuning_file = g_strdup("/home/reed/imx296-test/imx296_16mm.json");
    self->exposure_us = 8333;
    self->gain = 60;
    self->fd = -1;
    self->prop_zerocopy = TRUE;
    for (int i = 0; i < N_CAP_BUFS; i++) self->cap[i].fd = -1;
    isp_params_defaults(&self->isp_params);
    g_mutex_init(&self->lock);

    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
    gst_base_src_set_do_timestamp(GST_BASE_SRC(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(nvimx296_debug, "nvimx296camerasrc", 0,
                            "IMX296 CUDA ISP camera source");
    return gst_element_register(plugin, "nvimx296camerasrc", GST_RANK_PRIMARY,
                                gst_nvimx296camerasrc_get_type());
}

#define PACKAGE "nvimx296camerasrc"
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                  nvimx296camerasrc,
                  "IMX296 raw camera source with CUDA ISP",
                  plugin_init, "1.0", "GPL", "nvimx296camerasrc",
                  "https://github.com/peclatj/jetson-orin-nano-devkit-imx296-rpi-global-shutter")
