#include <iostream>
#include <sstream>
#include <vector>
#include <cublas_v2.h>


// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for GEMM
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/fp16_sgemm_traits.h"
#include "cutlass/gemm/hgemm_traits.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "cutlass/gemm/volta884_gemm_traits.h"
#include "cutlass/gemm/gemm.h"


#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
using namespace std;
#pragma warning( disable : 4503)

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

enum Layout {Column_major, Row_major};

#define CUDA_MODE
// #define TENSOR_MODE

#ifdef CUDA_MODE
typedef float type_ab;
typedef type_ab type_c;
#define CUDA_R CUDA_R_32F
#define CU_MATH CUBLAS_DEFAULT_MATH
#endif


#ifdef TENSOR_MODE
typedef half type_ab;
typedef type_ab type_c;
#define CUDA_R CUDA_R_16F
#define CU_MATH CUBLAS_TENSOR_OP_MATH
#endif

const int N_dim = 128;
const int MAX_BLOCK_NUM = 32;

// #define DEBUG

template<typename T_>
void print_matrix(const std::vector<T_> A, const int M, const int N, Layout lay, int num){
  printf("Matrix:");
  int i;
  float t; 
  int tn = 0;
  for(i = 0; i < M * N; i++){
    if(i % N == 0) printf("\n");
    int index = i;
    if(lay == Column_major){
      index = (i % N) * M + i / N;
    }
    t = A[index];
    printf("%0.3f\t", t);
    tn++;
    if(tn > num)
      return;
  }
  printf("\n");
  printf("\n");
}

template<typename T_>
void print_matrix(const T_* A, const int M, const int N, Layout lay, int num){
  printf("Matrix:");
  int i;
  float t; 
  int tn = 0;
  for(i = 0; i < M * N; i++){
    if(i % N == 0) printf("\n");
    int index = i;
    if(lay == Column_major){
      index = (i % N) * M + i / N;
    }
    t = A[index];
    printf("%0.3f\t", t);
    tn++;
    if(tn > num)
      return;
  }
  printf("\n");
  printf("\n");
}

template<typename T>
__global__ void float_to_half( 
    T*  d_array,
    float * s_array,
    const int n) {
    // Block index
    int bid_x = blockIdx.x;

    // Thread index
    int tid_x = threadIdx.x;
    
    int idx = bid_x * 256 + tid_x;
    if(idx > n)
      return;
    d_array[idx] = (T)s_array[idx];
}



template<typename T>
__global__ void deleteRow_kernel(
  T const *matrix,
  T *matrix_2,
  int rows,
  int columns,
  int *mask) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset_2 = i * columns + j;
    int offset = (i + mask[i])  * columns + j;
    matrix_2[offset_2] = matrix[offset] ;
  }
}

template<typename T>
__global__ void deleteColumn_kernel(
  T const *matrix,
  T *matrix_2,
  int rows,
  int columns,
  int *mask) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset_2 = i * columns + j;
    int offset = i * columns + j + mask[j];
    matrix_2[offset_2] = matrix[offset] ;
  }
}


template<typename T>
__global__ void insertRow_kernel(
  T *matrix,
  T const *matrix_2,
  int rows,
  int columns,
  int *mask) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset_2 = i * columns + j;
    int offset = (i + mask[i])  * columns + j;
    matrix[offset] = matrix_2[offset_2];
  }
}

template<typename T>
cudaError_t deleteRow(T const *matrix, T *matrix_2, int rows, int columns, int *mask) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  deleteRow_kernel<T><<< grid, block >>>(matrix, matrix_2, rows, columns, mask);

  return cudaGetLastError();
}

template<typename T>
cudaError_t deleteColumn(T const *matrix, T *matrix_2, int rows, int columns, int *mask) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  deleteColumn_kernel<T><<< grid, block >>>(matrix, matrix_2, rows, columns, mask);

  return cudaGetLastError();
}

template<typename T>
cudaError_t insertRow(T *matrix, T const *matrix_2, int rows, int columns, int *mask) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  insertRow_kernel<T><<< grid, block >>>(matrix, matrix_2, rows, columns, mask);

  return cudaGetLastError();
}
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUDA_MODE
typedef cutlass::gemm::SgemmTraits<
  cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
  cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
  cutlass::Shape<8, 128, 128>            // threadblock tile size
>
GemmTraits_NN;
#endif


#ifdef TENSOR_MODE
// typedef cutlass::gemm::Volta884GemmTraits<
//   cutlass::MatrixLayout::kColumnMajor,
//   cutlass::MatrixLayout::kColumnMajor,
//   cutlass::Shape<32, 128, 128>,
//   cutlass::Shape<32, 64, 64>,
//   half,
//   half,
//   half,
//   2
// > GemmTraits_NN;


  typedef cutlass::gemm::WmmaGemmTraits<
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<32, 128, 128>,
  type_ab,
  type_ab,
  type_c,
  cutlass::gemm::LinearScaling<half>,
  type_c,
  cutlass::Shape<32, 64, 32>
>
GemmTraits_NN;

#endif

// typedef cutlass::gemm::Volta884GemmTraits<
//   cutlass::MatrixLayout::kColumnMajor,
//   cutlass::MatrixLayout::kRowMajor,
//   cutlass::Shape<32, 128, 128>,
//   cutlass::Shape<32, 64, 32>,
//   half,
//   half,
//   half,
//   2
// > GemmTraits_NT;

// typedef cutlass::gemm::Volta884GemmTraits<
//   cutlass::MatrixLayout::kRowMajor,
//   cutlass::MatrixLayout::kColumnMajor,
//   cutlass::Shape<32, 128, 128>,
//   cutlass::Shape<32, 64, 32>,
//   half,
//   half,
//   half,
//   2
// > GemmTraits_TN;

// typedef cutlass::gemm::Volta884GemmTraits<
//   cutlass::MatrixLayout::kRowMajor,
//   cutlass::MatrixLayout::kRowMajor,
//   cutlass::Shape<32, 128, 128>,
//   cutlass::Shape<32, 64, 32>,
//   half,
//   half,
//   half,
//   2
// > GemmTraits_TT;

typedef cutlass::gemm::Gemm<GemmTraits_NN> Gemm_NN;
// typedef cutlass::gemm::Gemm<GemmTraits_NT> Gemm_NT;
// typedef cutlass::gemm::Gemm<GemmTraits_TN> Gemm_TN;
// typedef cutlass::gemm::Gemm<GemmTraits_TT> Gemm_TT;

cudaError_t Cutlass_Gemm_NN(
  int M,
  int N,
  int K,
  type_c alpha,
  type_ab const *A,
  int lda,
  type_ab const *B,
  int ldb,
  type_c beta,
  type_c *C,
  int ldc) {

  typename Gemm_NN::Params params;

  int result = params.initialize(
    M,     // GEMM M dimension
    N,     // GEMM N dimension
    K,     // GEMM K dimension
    alpha, // scalar alpha
    A,     // matrix A operand
    lda,
    B,     // matrix B operand
    ldb,
    beta,  // scalar beta
    C,     // source matrix C
    ldc,
    C,     // destination matrix C (may be different memory than source C matrix)
    ldc
  );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  cudaEvent_t startcutlass;
  cudaEvent_t stopcutlass;
  cudaEventCreate(&startcutlass);
  cudaEventCreate(&stopcutlass);
  cudaEventRecord(startcutlass);

  // Launch the CUTLASS GEMM kernel.
  int ites = 100;
  for (int ite = 0; ite < ites; ++ite) {
    Gemm_NN::launch(params);
    cudaDeviceSynchronize();
  }

  cudaEventRecord(stopcutlass);
  cudaEventSynchronize(stopcutlass);
  float cutlassTime;
  cudaEventElapsedTime(&cutlassTime, startcutlass, stopcutlass);
  cutlassTime /= (float)ites;

  printf("CUTLASS Baseline time %f ns.\n", cutlassTime * 1000);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t Stream_Gemm_NN(
  int M,
  int N,
  int K,
  type_c alpha,
  type_ab const *A,
  type_ab const *B,
  type_c beta,
  type_c *C,
  int block_num,
  int N_cal[],
  int K_cal[],
  int* mask_n[],
  int* mask_k[],
  int off_B[],
  int off_C[]) {
    
  typename Gemm_NN::Params params;  
  cudaStream_t stream[32];
  for(int i = 0; i < 32; i++)
  {
    cudaStreamCreate(&stream[i]);
  }

  cudaEvent_t startcutlass;
  cudaEvent_t stopcutlass;
  cudaEventCreate(&startcutlass);
  cudaEventCreate(&stopcutlass);
  cudaEventRecord(startcutlass);
  int ites = 100;
  for(int i = 0; i < ites; i++)
  {
    for(int bn = 0; bn < block_num; bn++ )
    {
      params.initialize(
        M,                // GEMM M dimension
        N_cal[bn],        // GEMM N dimension
        K_cal[bn],        // GEMM K dimension
        alpha,            // scalar alpha
        A,                // matrix A operand
        M,
        B + off_B[bn],    // matrix B operand
        K,
        beta,             // scalar beta
        C + off_C[bn],    // source matrix C
        M,
        C + off_C[bn],    // destination matrix C (may be different memory than source C matrix)
        M
      );
      Gemm_NN::launch(params, mask_k[bn], mask_n[bn], stream[bn]);
    }
    cudaDeviceSynchronize();
  }

  cudaEventRecord(stopcutlass);
  cudaEventSynchronize(stopcutlass);
  float cutlassTime;
  cudaEventElapsedTime(&cutlassTime, startcutlass, stopcutlass);
  cutlassTime /= (float)ites;

  printf("Stream GEMM time: %f ns\n", cutlassTime * 1000);
  for(int i = 0; i < 32; i ++)
  {
    cudaStreamDestroy(stream[i]);
  }
  return cudaGetLastError();
}
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t Stream_Gemm_NN_reference(
  int M,
  int N,
  int K,
  type_c alpha,
  type_ab const *A,
  type_ab const *B,
  type_c beta,
  type_c *C,
  int block_num,
  int N_cal[],
  int K_cal[],
  int* mask_n[],
  int* mask_k[],
  int off_B[],
  int off_C[]) {
    
  typename Gemm_NN::Params params;
  type_ab *A_2;
  cudaMalloc(&A_2, M * K * sizeof(type_ab));
  type_ab *C_2;
  cudaMalloc(&C_2, M * N * sizeof(type_c));

  for(int bn = 0; bn < block_num; bn++ )
  {
    deleteRow<type_ab>(A, A_2, K_cal[bn], M, mask_k[bn]);

    deleteRow<type_ab>(C + off_C[bn], C_2, N_cal[bn], M, mask_n[bn]);

    params.initialize(
      M,        // GEMM M dimension
      N_cal[bn],        // GEMM N dimension
      K_cal[bn],     // GEMM K dimension
      alpha,    // scalar alpha
      A_2,        // matrix A operand
      M,
      B + off_B[bn],     // matrix B operand
      K,
      beta,     // scalar beta
      C_2,        // source matrix C
      M,
      C_2,     // destination matrix C (may be different memory than source C matrix)
      M
    );
    Gemm_NN::launch(params);

    cudaDeviceSynchronize();

    insertRow<type_ab>(C + off_C[bn], C_2, N_cal[bn], M, mask_n[bn]);
  }
  cudaFree(A_2);
  cudaFree(C_2);
  return cudaGetLastError();
}
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t CuBlas_Gemm_NN_reference(
  int M,
  int N,
  int K,
  type_c alpha,
  type_ab const *A,
  type_ab const *B,
  type_c beta,
  type_c *C,
  int block_num,
  int N_cal[],
  int K_cal[],
  int* mask_n[],
  int* mask_k[],
  int off_B[],
  int off_C[]) {
    
  typename Gemm_NN::Params params;

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cublasSetMathMode(cublasHandle, CU_MATH);

  type_ab *A_2;
  cudaMalloc(&A_2, M * K * sizeof(type_ab));
  type_ab *C_2;
  cudaMalloc(&C_2, M * N * sizeof(type_c));

  for(int bn = 0; bn < block_num; bn++ )
  {
    deleteRow<type_ab>(A, A_2, K_cal[bn], M, mask_k[bn]);

    deleteRow<type_ab>(C + off_C[bn], C_2, N_cal[bn], M, mask_n[bn]);

    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      M, N_cal[bn], K_cal[bn],
      &alpha,
      A_2, CUDA_R, M,
      B + off_B[bn], CUDA_R, K,
      &beta,
      C_2, CUDA_R, M,
      CUDA_R, CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();

    insertRow<type_ab>(C + off_C[bn], C_2, N_cal[bn], M, mask_n[bn]);
  }
  cudaFree(A_2);
  cudaFree(C_2);
  return cudaGetLastError();
}
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  type_ab alpha,
  type_ab const *A,
  int lda,
  type_ab const *B,
  int ldb,
  type_ab beta,
  type_c *C,
  int ldc) {

  // Launch reference GEMM
  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;

  cudaEventCreate(&startcublas);
  cudaEventCreate(&stopcublas);


  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  // Use tensor cores
  //cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
  cublasSetMathMode(cublasHandle, CU_MATH);
  cudaEventRecord(startcublas);
  int ites = 100;
  for (int ite = 0; ite < ites; ++ite) {
  cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, CUDA_R, N,
                A, CUDA_R, K,
                &beta,
                C, CUDA_R, N,
                CUDA_R, CUBLAS_GEMM_DEFAULT);
  }
  cudaEventRecord(stopcublas);
  cudaEventSynchronize(stopcublas);
  float cublasTime;
  cudaEventElapsedTime(&cublasTime, startcublas, stopcublas);
  cublasTime /= (float)ites;
  printf("%f\n", cublasTime * 1000);

  return cudaGetLastError();
}

void init_matrix(float* mat, int M, int N)
{
  for(int i = 0; i < M; i++)
  {
    for(int j = 0; j < N; j++)
    {
      mat[i * N + j] = float( double (rand()) / double(RAND_MAX / 1) - 0.5 );
    }
  }
}
void init_matrix_zero(float* mat, int M, int N)
{
  for(int i = 0; i < M; i++)
  {
    for(int j = 0; j < N; j++)
    {
      mat[i * N + j] = 0.0;
    }
  }
}

void copy_matrix(float* dst, float* src, int length)
{
  for(int i = 0; i < length; i++)
  {
    dst[i] = src[i];
  }
}

void transpose_matrix(float* dst, float* src, int M, int N)
{
  for(int i = 0; i < M; i++)
  {
    for(int j = 0; j < N; j++)
    {
      dst[i * N + j] = src[j * M + i];
    }
  }
}


void gen_Ns(int N, int N_pruned, int K_ori[], int K_cal[], int b_num)
{
  if(b_num == 1)
  {
    K_ori[0] = N;
    K_cal[0] = N_pruned;
  }
  // Generate the K_ori K_cal;
  vector<int> mask_keep;
  vector<int> mask_host;
  for(int i = 0; i < N; i++)
  {
    mask_keep.push_back(i);
  }
  random_shuffle(mask_keep.begin(), mask_keep.end());

  for(int i = 0; i < N_pruned; i++)
  {
    mask_host.push_back(mask_keep[i]);
  }
  sort(mask_host.begin(), mask_host.end());
  
  K_ori[0] = mask_host[N_dim] + 1;
  K_cal[0] = N_dim;
  for(int i = 1; i < b_num - 1; i++)
  {
    K_ori[i] = mask_host[N_dim * i] - mask_host[N_dim * (i - 1)];
    K_cal[i] = N_dim;
  }
  K_ori[b_num - 1] = mask_host[N_pruned - 1] - mask_host[N_dim * (b_num - 1)];
  K_cal[b_num - 1] = N_pruned - N_dim * (b_num - 1);

  return;
}

void gen_Ks(int K, int K_pruned, int K_cal[], int b_num)
{
  // Generate the K_cal;
  for(int i = 0; i < b_num; i++)
  {
    K_cal[i] = K_pruned;
  }
}

vector<int> mask_gen(int N, int N_pruned)
{
  // Generate the mask;
  vector<int> mask_keep;
  vector<int> mask_host;
  for(int i = 0; i < N; i++)
  {
    mask_keep.push_back(i);
  }
  random_shuffle(mask_keep.begin(), mask_keep.end());  
  for(int i = 0; i < N_pruned; i++)
  {
    mask_host.push_back(mask_keep[i]);
  }
  sort(mask_host.begin(), mask_host.end());

  for(int i = 0; i < N_pruned; i++)
  {
    mask_host[i] = mask_host[i] - i;
  }
  return mask_host;
}

vector<int> mask_gen(int N, int N_pruned, int base)
{
  // Generate the mask_n;
  vector<int> mask_keep;
  vector<int> mask_host;
  for(int i = 0; i < N; i++)
  {
    mask_keep.push_back(i);
  }
  random_shuffle(mask_keep.begin(), mask_keep.end());  
  for(int i = 0; i < N_pruned; i++)
  {
    mask_host.push_back(mask_keep[i]);
  }
  sort(mask_host.begin(), mask_host.end());

  for(int i = 0; i < N_pruned; i++)
  {
    mask_host[i] = mask_host[i] - i + base;
  }
  return mask_host;
}

void gen_cutlass_weight(float* dst_mat, float* src_mat, int K, int N, vector<int> mask_k, vector<int> mask_n, int pruned_K, int pruned_N, int offset)
{
  for(int i = 0; i < pruned_K; i++)
  {
    for(int j = 0; j < pruned_N; j++)
    {
      int idx_col = mask_k[i] + i;
      int idx_row = mask_n[j] + j + offset;
      dst_mat[idx_col * N + idx_row] = src_mat[idx_col * N + idx_row];
    }
  }
  return;
}

void gen_masked_stream_weight(float* dst_mat, float* src_mat, int K, int N, vector<int> mask_k, vector<int> mask_n, int pruned_K, int pruned_N, int offset)
{
  for(int i = 0; i < pruned_K; i++)
  {
    for(int j = 0; j < pruned_N; j++)
    {
      int idx_col = mask_k[i] + i;
      int idx_row = mask_n[j] + j + offset;
      dst_mat[(j + offset) * K + i] = src_mat[idx_col * N + idx_row];
    }
  }
  return;
}

template<typename T>
void checkout(vector<T> host_cutlass, vector<T> host_reference, int M, int N, bool trans, double err)
{
  int errors = 0;
  for (int i = 0; i < M; i++) {
    for(int j = 0; j < N;  j++)
    {
      float v1 = 0.0;
      if(trans)
        v1 = host_reference[j * M + i];
      else
        v1 = host_reference[i * N + j];

      float v2 = host_cutlass[i * N + j];
      if (v1 / v2 > (1+err) || v2 / v1 > (1+err) || abs(v2 - v1) > err) {
        errors++;
        if (errors < 10) printf("%f %f\n", v1, v2);
      }        
    }
  }
  if (errors > 0) {
    std::cerr << "CUTLASS results incorrect. Errors : " << errors << " / "<<  (M*N) << std::endl << std::endl;
  }
  else
  {
    std::cerr << "PASSED !!!" << std::endl<< std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t TestCutlassGemm(int M, int N, int K, int N_pruned, int K_pruned, float alpha, float beta) {
  cudaError_t result;
  // Compute size in bytes of the C matrix.
  int sizeof_A = M * K;
  int sizeof_B = K * N;
  int sizeof_C = M * N;

  // Define pointers to matrices in host memory.
  float *hA_float = new float[sizeof_A];
  float *hA_float_transpose = new float[sizeof_A];
  
  float *hB_float = new float[sizeof_B];
  float *hB_float_purned = new float[sizeof_B];
  float *hB_float_purned_transpose = new float[sizeof_B];
  float *hB_float_stream = new float[sizeof_B];

  float *hC_float = new float[sizeof_C];
  float *hC_float_transpose = new float[sizeof_C];
  float *hC_float_stream = new float[sizeof_C];
  float *hC_float_reference = new float[sizeof_C];

  init_matrix(hA_float, M, K);
  transpose_matrix(hA_float_transpose, hA_float, K, M);

  init_matrix(hB_float, K, N);
  init_matrix_zero(hB_float_purned_transpose, K, N);
  init_matrix_zero(hB_float_purned, K, N);
  init_matrix_zero(hB_float_stream, K, N);

  init_matrix_zero(hC_float, M, N);
  init_matrix_zero(hC_float_transpose, M, N);
  init_matrix_zero(hC_float_stream, M, N);
  init_matrix_zero(hC_float_reference, M, N);


  int block_num = (N_pruned + N_dim - 1) / N_dim;
  assert(block_num <= MAX_BLOCK_NUM);
  printf("block_num : %d\n", block_num);

  // K_cal is pruned K size to calculate.
  int K_cal[MAX_BLOCK_NUM];
  for(int i = 0; i < block_num; i++)
  {
    K_cal[i] = K_pruned;
  }

  // N_ori is original N size of this block;
  int N_ori[MAX_BLOCK_NUM];
  // N_cal is pruned N size to calculate.
  int N_cal[MAX_BLOCK_NUM];
  int N_ori_off = 0;
  int N_cal_off = 0;
  for(int i = 0; i < block_num - 1; i++)
  {
    N_ori[i] = N / block_num;
    N_cal[i] = N_dim;
    N_ori_off += N_ori[i];
    N_cal_off += N_cal[i];
  }
  N_ori[block_num - 1] = N - N_ori_off;
  N_cal[block_num - 1] = N_pruned - N_cal_off;

  // for(int i = 0; i < block_num; i++)
  // {
  //   printf("%d %d\n", N_ori[i],N_cal[i]);
  // }

  // for(int i = 0; i < block_num; i++)
  // {
  //   printf("%d %d\n", K_cal[i],K_cal[i]);
  // }
  // gen_Ns(N, N_pruned, N_ori, N_cal, block_num);

  // To generate the global memory offset for B and C.
  int off_B[MAX_BLOCK_NUM];
  int off_C[MAX_BLOCK_NUM];
  int offset_B = 0;
  int offset_C = 0;
  for(int i = 0; i < block_num; i++)
  {
    off_B[i] = offset_B;
    off_C[i] = offset_C;
    offset_B += K * N_ori[i];
    offset_C += M * N_ori[i];
  }

  // To generate masks for K-dim and N-dim.
  int *mask_k[MAX_BLOCK_NUM];
  int *mask_n[MAX_BLOCK_NUM];

  for(int i = 0; i < block_num; i++)
  {
    vector<int> mask_k_host = mask_gen(K, K_cal[i]);
    cudaMalloc(&mask_k[i], K_cal[i] * sizeof(int));
    cudaMemcpy(mask_k[i], mask_k_host.data(), K_cal[i]*sizeof(int), cudaMemcpyHostToDevice);

    // for(int j = 0; j < mask_k_host.size(); j++)
    //  printf("%d ", mask_k_host[j]);
    //  printf("\n\n");

    vector<int> mask_n_host = mask_gen(N_ori[i], N_cal[i]);
    cudaMalloc(&mask_n[i], N_dim * sizeof(int));
    cudaMemcpy(mask_n[i], mask_n_host.data(), N_dim*sizeof(int), cudaMemcpyHostToDevice);

    // for(int j = 0; j < mask_n_host.size(); j++)
    //   printf("%d ", mask_n_host[j]);
    // printf(" \n\n");

    // Prune the weights
    gen_cutlass_weight(hB_float_purned, hB_float, K, N, mask_k_host, mask_n_host, K_cal[i], N_cal[i], off_C[i] / M);

    // Prune and organize weights for Stream GEMM
    gen_masked_stream_weight(hB_float_stream, hB_float, K, N, mask_k_host, mask_n_host, K_cal[i], N_cal[i], off_C[i] / M);
  }

  transpose_matrix(hB_float_purned_transpose, hB_float_purned, N, K);

  // print_matrix(hB_float, K, N, Row_major ,1024);
  // print_matrix(hB_float_purned_transpose, K, N, Column_major ,1024);
  // print_matrix(hB_float_stream, K, N, Column_major, 1024);

  // Define pointers to matrices in GPU device memory.
  float *dA_float;
  float *dA_float_transpose;

  float *dB_float_purned;
  float *dB_float_purned_transpose;
  float *dB_float_stream;

  float *dC_float;
  float *dC_float_transpose;
  float *dC_float_stream;
  float *dC_float_reference;

  int size_of_type = sizeof(float); 
  cudaMalloc(&dA_float, sizeof_A * size_of_type);
  cudaMalloc(&dA_float_transpose, sizeof_A * size_of_type);

  cudaMalloc(&dB_float_purned, sizeof_B * size_of_type);
  cudaMalloc(&dB_float_purned_transpose, sizeof_B * size_of_type);
  cudaMalloc(&dB_float_stream, sizeof_B * size_of_type);

  cudaMalloc(&dC_float, sizeof_C * size_of_type);
  cudaMalloc(&dC_float_transpose, sizeof_C * size_of_type);
  cudaMalloc(&dC_float_stream, sizeof_C * size_of_type);
  cudaMalloc(&dC_float_reference, sizeof_C * size_of_type);
  
  cudaMemcpy(dA_float, hA_float, sizeof_A * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dA_float_transpose, hA_float_transpose, sizeof_A * size_of_type, cudaMemcpyHostToDevice);

  cudaMemcpy(dB_float_purned, hB_float_purned, sizeof_B * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dB_float_purned_transpose, hB_float_purned_transpose, sizeof_B * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dB_float_stream, hB_float_stream, sizeof_B * size_of_type, cudaMemcpyHostToDevice);

  cudaMemcpy(dC_float, hC_float, sizeof_C * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dC_float_transpose, hC_float_transpose, sizeof_C * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dC_float_stream, hC_float_stream, sizeof_C * size_of_type, cudaMemcpyHostToDevice);
  cudaMemcpy(dC_float_reference, hC_float_reference, sizeof_C * size_of_type, cudaMemcpyHostToDevice);

  // Define pointers to matrices in GPU device memory using half.
  type_ab *A;
  type_ab *A_transpose;

  type_ab *B_pruned;
  type_ab *B_purned_transpose;
  type_ab *B_stream;

  type_c *C;
  type_c *C_transpose;
  type_c *C_stream;
  type_c *C_reference;

  size_of_type = sizeof(type_ab); 
  cudaMalloc(&A, sizeof_A * size_of_type);
  cudaMalloc(&A_transpose, sizeof_A * size_of_type);

  cudaMalloc(&B_pruned, sizeof_B * size_of_type);
  cudaMalloc(&B_purned_transpose, sizeof_B * size_of_type);
  cudaMalloc(&B_stream, sizeof_B * size_of_type);

  cudaMalloc(&C, sizeof_C * size_of_type);
  cudaMalloc(&C_transpose, sizeof_C * size_of_type);
  cudaMalloc(&C_stream, sizeof_C * size_of_type);
  cudaMalloc(&C_reference, sizeof_C * size_of_type);

  int block_size = 256;

  int grid_size = (sizeof_A + block_size - 1) / block_size;
  float_to_half<<<grid_size, block_size>>>(A, dA_float, sizeof_A);
  float_to_half<<<grid_size, block_size>>>(A_transpose, dA_float_transpose, sizeof_A);

  grid_size = (sizeof_B + block_size - 1) / block_size;
  float_to_half<<<grid_size, block_size>>>(B_pruned, dB_float_purned, sizeof_B);
  float_to_half<<<grid_size, block_size>>>(B_purned_transpose, dB_float_purned_transpose, sizeof_B);
  float_to_half<<<grid_size, block_size>>>(B_stream, dB_float_stream, sizeof_B);

  grid_size = (sizeof_C + block_size - 1) / block_size;
  float_to_half<<<grid_size, block_size>>>(C, dC_float, sizeof_C);
  float_to_half<<<grid_size, block_size>>>(C_transpose, dC_float_transpose, sizeof_C);
  float_to_half<<<grid_size, block_size>>>(C_stream, dC_float_stream, sizeof_C);
  float_to_half<<<grid_size, block_size>>>(C_reference, dC_float_reference, sizeof_C);


  // Cutlass Baseline
  result = Cutlass_Gemm_NN(N, M, K, alpha, B_pruned, N, A, K, beta, C, N);

  // Cutlass reference
  // result = Cutlass_Gemm_NN(M, N, K, alpha, A_transpose, M, B_purned_transpose, K, beta, C, M);
  result = Stream_Gemm_NN_reference(M, N, K, alpha, A_transpose, B_stream, beta, C_transpose, block_num, N_cal, K_cal, mask_n, mask_k, off_B, off_C);

  //Pruned Stream GEMM
  result = Stream_Gemm_NN(M, N, K, alpha, A_transpose, B_stream, beta, C_stream, block_num, N_cal, K_cal, mask_n, mask_k, off_B, off_C);

  //Pruned cuBlase GEMM
  // result = ReferenceGemm(M, N, K, alpha, A, N, B_pruned, K, beta, C_reference, N);
  result = CuBlas_Gemm_NN_reference(M, N, K, alpha, A_transpose, B_stream, beta, C_reference, block_num, N_cal, K_cal, mask_n, mask_k, off_B, off_C);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<type_c> host_cutlass(M * N, 0);
  std::vector<type_c> host_cutlass_transpose(M * N, 0);
  std::vector<type_c> host_stream(M * N, 0);
  std::vector<type_c> host_cublas(M * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C, sizeof_C * sizeof(type_c), cudaMemcpyDeviceToHost);
  result = cudaMemcpy(host_cutlass_transpose.data(), C_transpose, sizeof_C * sizeof(type_c), cudaMemcpyDeviceToHost);
  result = cudaMemcpy(host_stream.data(), C_stream, sizeof_C * sizeof(type_c), cudaMemcpyDeviceToHost);
  result = cudaMemcpy(host_cublas.data(), C_reference, sizeof_C * sizeof(type_c), cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;
      return result;
  }
 



#ifdef DEBUG
  std::vector<type_ab> host_A(sizeof_A, 0);
  std::vector<type_ab> host_A_trans(sizeof_A, 0);
  std::vector<type_ab> host_B(sizeof_B, 0);
  std::vector<type_ab> host_B_stream(sizeof_B, 0);

  cudaMemcpy(host_A.data(), A, sizeof_A * sizeof(type_ab), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_A_trans.data(), A_transpose, sizeof_A * sizeof(type_ab), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B.data(), B_pruned, sizeof_B * sizeof(type_ab), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B_stream.data(), B_stream, sizeof_B * sizeof(type_ab), cudaMemcpyDeviceToHost);

  print_matrix(host_A, M, K, Row_major,1024);
  print_matrix(host_A_trans, M, K, Column_major,1024);
  print_matrix(host_B, K, N, Row_major, 1024);
  print_matrix(host_B_stream, K, N, Column_major, 1024);

  print_matrix(host_cutlass, M, N, Row_major, 1024);
  print_matrix(host_cutlass_transpose, M, N, Column_major, 1024);
  print_matrix(host_stream, M, N, Column_major, 1024);
  print_matrix(host_cublas, M, N, Column_major, 1024);
#endif


  //The host_cutlass_transpose must be the exaly same with host_stream.
  checkout(host_cutlass_transpose, host_stream, M, N, false, 1e-10);

  //The host_cutlass is the CUTLASS baseline.
  checkout(host_cutlass, host_stream, M, N, true, 1e-5);

  //The CUTLASS source library has minor precision differences from cuBLAS.
  checkout(host_cutlass, host_cublas, M, N, true, 1e-3);

  //
  // Free device memory allocations.
  //
  delete[] hA_float;
  delete[] hA_float_transpose;
  delete[] hB_float;
  delete[] hB_float_purned;
  delete[] hB_float_purned_transpose;
  delete[] hB_float_stream;
  delete[] hC_float;
  delete[] hC_float_transpose;
  delete[] hC_float_stream;
  delete[] hC_float_reference;

  cudaFree(dA_float);
  cudaFree(dA_float_transpose);
  cudaFree(dB_float_purned);
  cudaFree(dB_float_purned_transpose);
  cudaFree(dB_float_stream);
  cudaFree(dC_float);
  cudaFree(dC_float_transpose);
  cudaFree(dC_float_stream);
  cudaFree(dC_float_reference);
  cudaFree(A);
  cudaFree(A_transpose);
  cudaFree(B_pruned);
  cudaFree(B_purned_transpose);
  cudaFree(B_stream);
  cudaFree(C);
  cudaFree(C_transpose);
  cudaFree(C_stream);
  cudaFree(C_reference);
  for(int i = 0; i < block_num; i++)
  {
    cudaFree(mask_k[i]);
    cudaFree(mask_n[i]);
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to stream_gemm example.
//
// usage:
//
//   test_basic_nn <M> <N> <K> <N_pruned> <K_pruned>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions.
  //

  // GEMM problem dimensions.
  // M N K N_pruned K_pruned
  int problem[5] = {768, 768, 768, 768, 768};

  for (int i = 1; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }
  
  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  //
  // Run the CUTLASS GEMM test.
  //
  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    problem[3],     // GEMM N_pruned dimension
    problem[4],     // GEMM K_pruned dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}