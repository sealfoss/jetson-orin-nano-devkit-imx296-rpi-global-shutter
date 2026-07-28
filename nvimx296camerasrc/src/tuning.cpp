/* SPDX-License-Identifier: GPL-2.0-only
 * Minimal targeted parser for the RPi libcamera imx296 tuning JSON
 * (imx296_16mm.json). Not a general JSON parser: it extracts exactly the
 * four blocks the ISP needs (rpi.black_level, rpi.ccm, rpi.awb ct_curve,
 * rpi.contrast gamma_curve) from the known file structure, and fails
 * loudly on anything unexpected. json-glib is not present on L4T.
 */
#include "imx296_isp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(*len + 1);
    if (!buf || fread(buf, 1, *len, f) != (size_t)*len) { fclose(f); free(buf); return NULL; }
    buf[*len] = 0; fclose(f);
    return buf;
}

/* Parse up to max doubles from the JSON array starting at the first '['
 * at or after p. Returns count, or -1. */
static int parse_num_array(const char *p, double *out, int max)
{
    const char *q = strchr(p, '[');
    if (!q) return -1;
    q++;
    int n = 0;
    while (n < max) {
        char *end;
        double v = strtod(q, &end);
        if (end == q) break;
        out[n++] = v;
        q = end;
        while (*q == ',' || *q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') q++;
        if (*q == ']') return n;
    }
    return (*q == ']') ? n : -1;  /* array bigger than max -> -1 */
}

static double parse_num_after(const char *p, const char *key, int *ok)
{
    const char *q = strstr(p, key);
    if (!q) { *ok = 0; return 0; }
    q = strchr(q + strlen(key), ':');
    if (!q) { *ok = 0; return 0; }
    char *end; double v = strtod(q + 1, &end);
    *ok = (end != q + 1);
    return v;
}

int isp_tuning_load(const char *path, IspTuning *t)
{
    long len = 0;
    char *buf = slurp(path, &len);
    if (!buf) { fprintf(stderr, "tuning: cannot read %s\n", path); return -1; }
    memset(t, 0, sizeof(*t));
    int ok = 0;

    /* black level */
    const char *bl = strstr(buf, "\"rpi.black_level\"");
    if (!bl) goto fail;
    t->black_level_16bit = parse_num_after(bl, "\"black_level\"", &ok);
    if (!ok || t->black_level_16bit <= 0 || t->black_level_16bit >= 65536) goto fail;

    /* ccms: sequence of { "ct": N, "ccm": [9 numbers] }, bounded by the
     * ccms array's own closing bracket (block order in the file varies). */
    {
        const char *p = strstr(buf, "\"rpi.ccm\"");
        if (!p) goto fail;
        const char *ccms = strstr(p, "\"ccms\"");
        if (!ccms) goto fail;
        const char *arr = strchr(ccms, '[');
        if (!arr) goto fail;
        /* find matching ] by depth scan */
        const char *end = arr; int depth = 0;
        do {
            if (*end == '[') depth++;
            else if (*end == ']') depth--;
            end++;
        } while (depth > 0 && *end);
        const char *q = arr;
        while (t->n_ccms < ISP_MAX_CCMS && q < end) {
            const char *ct = strstr(q, "\"ct\"");
            if (!ct || ct >= end) break;
            const char *ccm = strstr(ct, "\"ccm\"");
            if (!ccm || ccm >= end) break;
            double ctv = parse_num_after(ct, "\"ct\"", &ok);
            if (!ok) break;
            double m[9];
            if (parse_num_array(ccm, m, 9) != 9) break;
            t->ccm_ct[t->n_ccms] = ctv;
            memcpy(t->ccm[t->n_ccms], m, sizeof(m));
            t->n_ccms++;
            q = strchr(ccm, ']');           /* past this entry's ccm array */
            if (!q) break;
        }
        if (t->n_ccms < 2) goto fail;
    }

    /* awb ct_curve: flat triples */
    {
        const char *p = strstr(buf, "\"rpi.awb\"");
        if (!p) goto fail;
        const char *cc = strstr(p, "\"ct_curve\"");
        if (!cc) goto fail;
        double flat[ISP_MAX_CT_POINTS * 3];
        int n = parse_num_array(cc, flat, ISP_MAX_CT_POINTS * 3);
        if (n < 6 || n % 3) goto fail;
        t->n_ct = n / 3;
        for (int i = 0; i < t->n_ct; i++) {
            t->ct_curve[i][0] = flat[i * 3];
            t->ct_curve[i][1] = flat[i * 3 + 1];
            t->ct_curve[i][2] = flat[i * 3 + 2];
        }
    }

    /* contrast gamma_curve: flat x,y pairs in 16-bit domain */
    {
        const char *p = strstr(buf, "\"rpi.contrast\"");
        if (!p) goto fail;
        const char *gc = strstr(p, "\"gamma_curve\"");
        if (!gc) goto fail;
        double flat[ISP_GAMMA_MAX_POINTS * 2];
        int n = parse_num_array(gc, flat, ISP_GAMMA_MAX_POINTS * 2);
        if (n < 4 || n % 2) goto fail;
        t->n_gamma = n / 2;
        for (int i = 0; i < t->n_gamma; i++) {
            t->gamma_x[i] = flat[i * 2] / 65535.0;
            t->gamma_y[i] = flat[i * 2 + 1] / 65535.0;
        }
    }

    free(buf);
    return 0;
fail:
    fprintf(stderr, "tuning: %s does not match the expected imx296 tuning layout\n", path);
    free(buf);
    return -1;
}
