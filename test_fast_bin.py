import pstats, cProfile

import pyximport
pyximport.install()

import line_profiler

import numpy as np

from fast_binning import fast_bin

@line_profiler.profile
def main():
    A = np.arange(8*8).reshape(8,8).astype(np.uint8)


    res = fast_bin(A, 4, np.dtype(np.uint16).num)

    print(res)
    print(res.dtype)

if __name__ == "__main__":

    main()

    # cProfile.runctx("main()", globals(), locals(), "mytest.prof")
    # s = pstats.Stats("mytest.prof")
    # s.strip_dirs().sort_stats("time").print_stats()




