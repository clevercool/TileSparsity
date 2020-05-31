
#pragma once
#include "cutlass/gemm/wmma_gemm_traits.h"

template <
int s1,
int s2,
int s3,
int s4,
int s5,
int s6>
class WMMA_C_Gemm
{
public:
  typedef cutlass::gemm::WmmaGemmTraits<
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kRowMajor,
  cutlass::Shape<s1, s2, s3>,
  half,
  half,
  half,
  cutlass::gemm::LinearScaling<half>,
  half, 
  cutlass::Shape<s4, s5, s6>>
  GemmTraits;
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
};

template <std::size_t type>
class WMMA_Gemm;

//Fastest
template <>
class WMMA_Gemm<0>
{
public:
  typedef WMMA_C_Gemm<32, 256, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class WMMA_Gemm<1>
{
public:
  typedef WMMA_C_Gemm<64, 256, 128, 64, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class WMMA_Gemm<2>
{
public:
  typedef WMMA_C_Gemm<32, 128, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};


template <>
class WMMA_Gemm<3>
{
public:
  typedef WMMA_C_Gemm<32, 128, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class WMMA_Gemm<4>
{
public:
  typedef WMMA_C_Gemm<32, 256, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};

template <>
class WMMA_Gemm<5>
{
public:
  typedef WMMA_C_Gemm<32, 128, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};


template <>
class WMMA_Gemm<6>
{
public:
  typedef WMMA_C_Gemm<32, 256, 128, 32, 64, 64> cutlass_gemm;
  typedef cutlass_gemm::Gemm Gemm;
};