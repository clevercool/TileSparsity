
#pragma once
#include "cutlass/gemm/volta884_gemm_traits.h"
#include "cutlass/gemm/gemm.h"

template <
int s1,
int s2,
int s3,
int s4 = 32,
int s5 = 64,
int s6 = 64,
int s7 = 16,
int s8 = 16,
int s9 = 16>
class H884_C_Gemm
{
public:

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<s1, s2, s3>,
    cutlass::Shape<s4, s5, s6>,
    half,
    half,
    half,
    2
  > 
  GemmTraits;
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
};

template <std::size_t type>
class H884_Gemm;

// Fastest
template <>
class H884_Gemm<0>
{
public:
  typedef H884_C_Gemm<32, 128, 128> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class H884_Gemm<1>
{
public:
  typedef H884_C_Gemm<32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class H884_Gemm<2>
{
public:
  typedef H884_C_Gemm<32, 256, 128> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};


template <>
class H884_Gemm<3>
{
public:
  typedef H884_C_Gemm<32, 256, 128> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class H884_Gemm<4>
{
public:
  typedef H884_C_Gemm<32, 256, 128, 32, 64, 64, 16, 8, 32> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class H884_Gemm<5>
{
public:
  typedef H884_C_Gemm<32, 256, 128, 32, 64, 64, 16, 32, 8> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};
