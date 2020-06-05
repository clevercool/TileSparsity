Stream masked CUTLASS
===================

Profiling for SC 2020.

## Build and Run
```shell
$ git clone https://github.com/clevercool/SCTest.git
$ cd SCTest
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cd perf
$ make

$ #Base GEMM size for BERT
$ #./test_basic_nn <M> <N> <K> <N_pruned> <K_pruned>
$ # 75% Sparsity Test
$ ./test_basic_nn 12800 768 768 384 384        # 4 times.
CUTLASS GEMM : 252.878403 us.
CUTLASS GEMM : 252.665283 us.
Stream  GEMM : 118.173439 us.

$ ./test_basic_nn 12800 768 3072 384 1536
CUTLASS GEMM : 874.368286 us.
CUTLASS GEMM : 831.650879 us.
Stream  GEMM : 298.622406 us.

$ ./test_basic_nn 12800 3072 768 1536 384
CUTLASS GEMM : 920.689941 us.
CUTLASS GEMM : 849.862732 us.
Stream  GEMM : 417.427826 us.
```

Stream GEMM has about 2.26x speedup than CUTLASS GEMM with 75% sparsity.