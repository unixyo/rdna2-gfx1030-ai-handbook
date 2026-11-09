/* Research-only MIOpen 3.6 NORMAL Find workspace correction.
 *
 * 8d1ae90e convolution.cpp excludes GEMM workspace whenever dilation > 1,
 * although the corresponding GEMM solvers can run those convolutions.
 * This interposes only the three workspace queries and uses the installed
 * runtime's exported Range API to include every applicable solver's request.
 * No convolution, optimizer, tensor, or installed library is replaced.
 *
 * Enabled only by APPLIO_RDNA2_FIX_FIND_WORKSPACE=1 with MIOPEN_FIND_MODE=NORMAL
 * (or 1). FAST replay and all other processes use the original query result.
 * Build: cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror \
 *          v08_miopen_workspace.c -ldl -o <research-path>/libv08_miopen_workspace.so
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MIOpen C ABI: status is an enum (int), descriptors/handle are opaque pointers.
 * Signatures are recorded in source/miopen_workspace/convolution_api.cpp from
 * the same commit as the installed MIOpen database key, 8d1ae90e.
 */
typedef int (*workspace_query)(void *, const void *, const void *,
                               const void *, const void *, size_t *);
typedef int (*workspace_range)(void *, const void *, const void *,
                               const void *, const void *, size_t *, size_t *);

enum { V08_BAD_PARAMETER = 3, V08_INTERNAL_ERROR = 5 };

static int enabled(void)
{
    const char *flag = getenv("APPLIO_RDNA2_FIX_FIND_WORKSPACE");
    const char *mode = getenv("MIOPEN_FIND_MODE");
    return flag && strcmp(flag, "1") == 0 && mode &&
           (strcmp(mode, "NORMAL") == 0 || strcmp(mode, "1") == 0);
}

static int corrected_query(const char *original_name, const char *range_name,
                           const char *direction, void *handle,
                           const void *first, const void *second,
                           const void *conv, const void *last, size_t *bytes)
{
    if (!bytes)
        return V08_BAD_PARAMETER;
    workspace_query original = (workspace_query)dlsym(RTLD_NEXT, original_name);
    if (!original) {
        fprintf(stderr, "V08_WORKSPACE_ERROR %s missing_original_symbol\n", direction);
        return V08_INTERNAL_ERROR;
    }
    int status = original(handle, first, second, conv, last, bytes);
    if (status || !enabled())
        return status;

    workspace_range range = (workspace_range)dlsym(RTLD_NEXT, range_name);
    if (!range) {
        fprintf(stderr, "V08_WORKSPACE_ERROR %s missing_range_symbol\n", direction);
        return V08_INTERNAL_ERROR;
    }
    const size_t legacy = *bytes;
    size_t minimum = 0, maximum = 0;
    status = range(handle, first, second, conv, last, &minimum, &maximum);
    if (status) {
        fprintf(stderr, "V08_WORKSPACE_ERROR %s range_status=%d legacy=%zu\n",
                direction, status, legacy);
        return status;
    }
    if (minimum > maximum) {
        fprintf(stderr, "V08_WORKSPACE_ERROR %s invalid_range=%zu:%zu\n",
                direction, minimum, maximum);
        return V08_INTERNAL_ERROR;
    }
    if (maximum > *bytes)
        *bytes = maximum;
    fprintf(stderr,
            "V08_WORKSPACE direction=%s legacy_bytes=%zu range_min_bytes=%zu "
            "range_max_bytes=%zu returned_bytes=%zu\n",
            direction, legacy, minimum, maximum, *bytes);
    return 0;
}

int miopenConvolutionForwardGetWorkSpaceSize(void *handle, const void *w,
                                            const void *x, const void *conv,
                                            const void *y, size_t *bytes)
{
    return corrected_query("miopenConvolutionForwardGetWorkSpaceSize",
                           "miopenConvolutionForwardGetWorkSpaceSizeRange",
                           "fwd", handle, w, x, conv, y, bytes);
}

int miopenConvolutionBackwardDataGetWorkSpaceSize(void *handle, const void *dy,
                                                 const void *w, const void *conv,
                                                 const void *dx, size_t *bytes)
{
    return corrected_query("miopenConvolutionBackwardDataGetWorkSpaceSize",
                           "miopenConvolutionBackwardDataGetWorkSpaceSizeRange",
                           "dx", handle, dy, w, conv, dx, bytes);
}

int miopenConvolutionBackwardWeightsGetWorkSpaceSize(void *handle, const void *dy,
                                                    const void *x, const void *conv,
                                                    const void *dw, size_t *bytes)
{
    return corrected_query("miopenConvolutionBackwardWeightsGetWorkSpaceSize",
                           "miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange",
                           "dw", handle, dy, x, conv, dw, bytes);
}
