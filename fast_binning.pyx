#cython: language_level=3
#cython: profile=False
#distutils: extra_compile_args=-fopenmp
#distutils: extra_link_args=-fopenmp
cimport numpy as cnp
cimport cython
cimport openmp
from cpython cimport PyObject
from cython.parallel import prange

# essential for setting up NumPy-C API
cnp.import_array()

ctypedef fused uintegral:
    cython.uchar
    cython.ushort
    cython.uint
    cython.ulong

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_2(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 2
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_3(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 3
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_4(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 4
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] + arr_v[row,   col+3] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] + arr_v[row+1, col+3] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2] + arr_v[row+2, col+3] +\
                                 arr_v[row+3, col] + arr_v[row+3, col+1] + arr_v[row+3, col+2] + arr_v[row+3, col+3]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_5(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 5
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] + arr_v[row,   col+3] + arr_v[row,   col+4] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] + arr_v[row+1, col+3] + arr_v[row+1, col+4] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2] + arr_v[row+2, col+3] + arr_v[row+2, col+4] +\
                                 arr_v[row+3, col] + arr_v[row+3, col+1] + arr_v[row+3, col+2] + arr_v[row+3, col+3] + arr_v[row+3, col+4] +\
                                 arr_v[row+4, col] + arr_v[row+4, col+1] + arr_v[row+4, col+2] + arr_v[row+4, col+3] + arr_v[row+4, col+4]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_6(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 6
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] + arr_v[row,   col+3] + arr_v[row,   col+4] + arr_v[row,   col+5] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] + arr_v[row+1, col+3] + arr_v[row+1, col+4] + arr_v[row+1, col+5] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2] + arr_v[row+2, col+3] + arr_v[row+2, col+4] + arr_v[row+2, col+5] +\
                                 arr_v[row+3, col] + arr_v[row+3, col+1] + arr_v[row+3, col+2] + arr_v[row+3, col+3] + arr_v[row+3, col+4] + arr_v[row+3, col+5] +\
                                 arr_v[row+4, col] + arr_v[row+4, col+1] + arr_v[row+4, col+2] + arr_v[row+4, col+3] + arr_v[row+4, col+4] + arr_v[row+4, col+5] +\
                                 arr_v[row+5, col] + arr_v[row+5, col+1] + arr_v[row+5, col+2] + arr_v[row+5, col+3] + arr_v[row+5, col+4] + arr_v[row+5, col+5]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_7(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 7
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] + arr_v[row,   col+3] + arr_v[row,   col+4] + arr_v[row,   col+5] + arr_v[row,   col+6] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] + arr_v[row+1, col+3] + arr_v[row+1, col+4] + arr_v[row+1, col+5] + arr_v[row+1, col+6] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2] + arr_v[row+2, col+3] + arr_v[row+2, col+4] + arr_v[row+2, col+5] + arr_v[row+2, col+6] +\
                                 arr_v[row+3, col] + arr_v[row+3, col+1] + arr_v[row+3, col+2] + arr_v[row+3, col+3] + arr_v[row+3, col+4] + arr_v[row+3, col+5] + arr_v[row+3, col+6] +\
                                 arr_v[row+4, col] + arr_v[row+4, col+1] + arr_v[row+4, col+2] + arr_v[row+4, col+3] + arr_v[row+4, col+4] + arr_v[row+4, col+5] + arr_v[row+4, col+6] +\
                                 arr_v[row+5, col] + arr_v[row+5, col+1] + arr_v[row+5, col+2] + arr_v[row+5, col+3] + arr_v[row+5, col+4] + arr_v[row+5, col+5] + arr_v[row+5, col+6] +\
                                 arr_v[row+6, col] + arr_v[row+6, col+1] + arr_v[row+6, col+2] + arr_v[row+6, col+3] + arr_v[row+6, col+4] + arr_v[row+6, col+5] + arr_v[row+6, col+6]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _bin_by_8(uintegral[:, ::1] arr_v,
                    cython.ulong[:, ::1] binned_v,
                    unsigned long int nc,
                    unsigned long int bnr,
                    unsigned long int bnc
                    ):
    cdef:
        int bw = 8
        unsigned long int bsz = bnr * bnc
        unsigned long int k = 0
        unsigned long int corner = 0
        unsigned long int row = 0
        unsigned long int col = 0
        unsigned long int scol = bw*nc

    for k in prange(bsz, nogil=True, schedule="static"):
        corner = (k*bw)%nc + (k/bnc)*scol
        row = corner / nc
        col = corner % nc

        binned_v[k/bnc, k%bnc] = arr_v[row,   col] + arr_v[row,   col+1] + arr_v[row,   col+2] + arr_v[row,   col+3] + arr_v[row,   col+4] + arr_v[row,   col+5] + arr_v[row,   col+6] + arr_v[row,   col+7] +\
                                 arr_v[row+1, col] + arr_v[row+1, col+1] + arr_v[row+1, col+2] + arr_v[row+1, col+3] + arr_v[row+1, col+4] + arr_v[row+1, col+5] + arr_v[row+1, col+6] + arr_v[row+1, col+7] +\
                                 arr_v[row+2, col] + arr_v[row+2, col+1] + arr_v[row+2, col+2] + arr_v[row+2, col+3] + arr_v[row+2, col+4] + arr_v[row+2, col+5] + arr_v[row+2, col+6] + arr_v[row+2, col+7] +\
                                 arr_v[row+3, col] + arr_v[row+3, col+1] + arr_v[row+3, col+2] + arr_v[row+3, col+3] + arr_v[row+3, col+4] + arr_v[row+3, col+5] + arr_v[row+3, col+6] + arr_v[row+3, col+7] +\
                                 arr_v[row+4, col] + arr_v[row+4, col+1] + arr_v[row+4, col+2] + arr_v[row+4, col+3] + arr_v[row+4, col+4] + arr_v[row+4, col+5] + arr_v[row+4, col+6] + arr_v[row+4, col+7] +\
                                 arr_v[row+5, col] + arr_v[row+5, col+1] + arr_v[row+5, col+2] + arr_v[row+5, col+3] + arr_v[row+5, col+4] + arr_v[row+5, col+5] + arr_v[row+5, col+6] + arr_v[row+5, col+7] +\
                                 arr_v[row+6, col] + arr_v[row+6, col+1] + arr_v[row+6, col+2] + arr_v[row+6, col+3] + arr_v[row+6, col+4] + arr_v[row+6, col+5] + arr_v[row+6, col+6] + arr_v[row+6, col+7] +\
                                 arr_v[row+7, col] + arr_v[row+7, col+1] + arr_v[row+7, col+2] + arr_v[row+7, col+3] + arr_v[row+7, col+4] + arr_v[row+7, col+5] + arr_v[row+7, col+6] + arr_v[row+7, col+7]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def fast_bin(uintegral[:, ::1] arr, int bw, int typenum):
    """Fast 2D binning of array with unsigned integer data type.

    Parameter
    ---------
    arr : 2D memoryview
        the memory view of an object supporting array buffer protocol.
        This must be C-contiguous.
    bw : int
        the bin width, and it must be between 2 and 8 inclusive.
    typenum : int
        the unique data dtype number for an unsigned integer data type.
        This is the data dtype of the binned array.

    Returns
    -------
    binned : ndarray
        the binned array
    """
    cdef:
        unsigned long int nr = arr.shape[0]
        unsigned long int nc = arr.shape[1]
        unsigned long int bnr = nr / bw
        unsigned long int bnc = nc / bw
        cnp.npy_intp[2] bin_dims = [bnr, bnc]

    if (nr % bw != 0) or (nc % bw != 0):
        msg = (f"The number of row ({nr}) and column ({nc}) of the 2D array "
               f"must be divisible by the bin width ({bw}).")
        raise ValueError(msg)

    if (bw <= 1) or (bw > 8):
        msg = f"The bin width must be 2 <= bw <= 8, but {bw} is provided."
        raise ValueError(msg)

    # regardless of the input data type, use np.uint64 to hold the
    # binned value
    binned = cnp.PyArray_EMPTY(2, bin_dims, cnp.NPY_UINT64, 0)
    cdef cython.ulong[:, ::1] binned_v = binned

    if (bw == 2):
        _bin_by_2(arr, binned_v, nc, bnr, bnc)
    elif (bw == 3):
        _bin_by_3(arr, binned_v, nc, bnr, bnc)
    elif (bw == 4):
        _bin_by_4(arr, binned_v, nc, bnr, bnc)
    elif (bw == 5):
        _bin_by_5(arr, binned_v, nc, bnr, bnc)
    elif (bw == 6):
        _bin_by_6(arr, binned_v, nc, bnr, bnc)
    elif (bw == 7):
        _bin_by_7(arr, binned_v, nc, bnr, bnc)
    elif (bw == 8):
        _bin_by_8(arr, binned_v, nc, bnr, bnc)

    if typenum == cnp.NPY_UINT8:
        return cnp.PyArray_Cast(binned, cnp.NPY_UINT8)

    if typenum == cnp.NPY_UINT16:
        return cnp.PyArray_Cast(binned, cnp.NPY_UINT16)

    if typenum == cnp.NPY_UINT32:
        return cnp.PyArray_Cast(binned, cnp.NPY_UINT32)

    return binned



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def bincrop_2D(uintegral[:, ::1] arr, int bw):
    # cdef:
        # unsigned long int nr = arr.shape[0]
        # unsigned long int nc = arr.shape[1]
        # unsigned long int sz = nr*nc
        # unsigned long int bnr = nr / bw
        # unsigned long int bnc = nc / bw
        # unsigned long int bsz = bnr * bnc
        # int nthreads = openmp.omp_get_max_threads()
        # cnp.npy_intp tbufsz = bsz * nthreads
        # cnp.npy_intp[2] bin_dims = [bnr, bnc]
        # unsigned long int row_shift = nr % bw
        # unsigned long int col_shift = nc % bw
        # unsigned long int rc = 0
        # unsigned long int brc = 0
        # int binned_row_idx, binned_col_idx, norm_rc, tid

    # # if num_threads == -1:
    # # else:
        # # cdef int nthreads = num_threads

    # # cdef openmp.omp_lock_t lock
    # # cdef uintegral total=0

    # if uintegral is cython.uchar:
        # binned = cnp.PyArray_ZEROS(2, bin_dims, cnp.NPY_UINT8, 0)
        # t_buf = cnp.PyArray_ZEROS(1, &tbufsz, cnp.NPY_UINT8, 0)
    # elif uintegral is cython.ushort:
        # binned = cnp.PyArray_ZEROS(2, bin_dims, cnp.NPY_UINT16, 0)
        # t_buf = cnp.PyArray_ZEROS(1, &tbufsz, cnp.NPY_UINT16, 0)
    # elif uintegral is cython.uint:
        # binned = cnp.PyArray_ZEROS(2, bin_dims, cnp.NPY_UINT32, 0)
        # t_buf = cnp.PyArray_ZEROS(1, &tbufsz, cnp.NPY_UINT32, 0)
    # elif uintegral is cython.ulong:
        # binned = cnp.PyArray_ZEROS(2, bin_dims, cnp.NPY_UINT64, 0)
        # t_buf = cnp.PyArray_ZEROS(1, &tbufsz, cnp.NPY_UINT64, 0)

    # cdef uintegral[:, ::1] binned_v = binned
    # cdef uintegral[::1] t_buf_v = t_buf

    # # num_threads = openmp.omp_get_num_threads()
    # # num_threads = openmp.omp_get_max_threads()

    # for rc in prange(sz, nogil=True):
        # if ((rc / nc >= row_shift) and (rc % nc >= col_shift)):
            # tid = openmp.omp_get_thread_num()

            # norm_rc = rc - col_shift - nc*row_shift

            # binned_row_idx = norm_rc / (nc*bw)
            # binned_col_idx = (norm_rc % nc) / bw

            # # avoid this race condition!
            # # binned_v[binned_row_idx, binned_col_idx] += arr[rc / nc, rc % nc]

            # # Z_local[tid * nbins_padded + b] += Xij*Yij
            # t_buf_v[(binned_row_idx*bnc + binned_col_idx)*nthreads + tid] += arr[rc / nc, rc % nc]

    # # sum the values in the thread buffer
    # for brc in range(bsz):
        # for tid in range(nthreads):
            # binned_v[brc / bnc, brc % bnr] += t_buf_v[brc*nthreads + tid]

    # return binned
