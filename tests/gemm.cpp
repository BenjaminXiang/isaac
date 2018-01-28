#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cfenv>

#include "isaac/driver/backend.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/gemm.h"
#include "isaac/templates/error.hpp"
#include "isaac/tools/collections.hpp"

#include "isaac/api.h"

#include "test_utils.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

template<char> inline int matidx(int i, int j, int ld);
template<> inline int matidx<'T'>(int i, int j, int ld){ return j + i*ld; }
template<> inline int matidx<'N'>(int i, int j, int ld){ return i + j*ld; }

template<class T, char AT, char BT>
void cpp_gemm_impl(int M, int N, int K, T* C, int ldc, T alpha, T* A, int lda, T* B, int ldb, T beta){
  for(int i = 0; i < M; ++i)
    for(int j = 0; j < N; ++j){
      T acc = 0;
      for(int k = 0; k < K; ++k)
        acc += A[matidx<AT>(i, k, lda)]*B[matidx<BT>(k, j, ldb)];
      C[i + j*ldc] = alpha*acc + ((beta!=0)?beta*C[i + j*ldc]:0);
  }
}

template<class T>
void cpp_gemm(int M, int N, int K, T* C, int ldc, T alpha, T* A, int lda, T* B, int ldb, T beta, char AT, char BT){
  if(AT=='N' && BT=='N') cpp_gemm_impl<T, 'N','N'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta);
  if(AT=='T' && BT=='N') cpp_gemm_impl<T, 'T','N'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta);
  if(AT=='N' && BT=='T') cpp_gemm_impl<T, 'N','T'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta);
  if(AT=='T' && BT=='T') cpp_gemm_impl<T, 'T','T'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta);
}

template<class DTYPE>
void do_test(sc::driver::Context const & ctx, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, int32_t M, int32_t N, int32_t K){
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);

  //Shapes
  int32_t AS0 = M, AS1 = K;
  int32_t BS0 = K, BS1 = N;
  if(AT==sc::ISAAC_OP_T) std::swap(AS0, AS1);
  if(BT==sc::ISAAC_OP_T) std::swap(BS0, BS1);
  int32_t ldc = M, lda = AS0, ldb = BS0;
  int32_t offc = 0, offa = 0, offb = 0;
  DTYPE alpha = 1., beta = 3.2;
  sc::scalar sc_alpha(alpha, dtype), sc_beta(beta, dtype);

  //Initialize Buffers
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  std::vector<DTYPE> iC(M*N);
  std::vector<DTYPE> iA(M*K);
  std::vector<DTYPE> iB(K*N);
  std::vector<DTYPE> rC(M*N);
  srand(0);
  for(size_t i = 0; i < iA.size(); ++i) iA[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < iB.size(); ++i) iB[i] = (float)rand()/RAND_MAX;

  drv::Stream stream(ctx);
  stream.write(C, true, 0, M*N*dtsize, iC.data());
  stream.write(A, true, 0, M*K*dtsize, iA.data());
  stream.write(B, true, 0, K*N*dtsize, iB.data());

  //Ground truth
  char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
  char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';
  cpp_gemm<DTYPE>(M, N, K, rC.data(), ldc, (DTYPE)alpha, iA.data(), lda, iB.data(), ldb, (DTYPE)beta, cuAT, cuBT);

  //ISAAC results
  std::vector<DTYPE> hC(M*N);
  sc::GEMM(ctx.device(), stream, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, sc_alpha, A, B, sc_beta, C);
  stream.read(C, true, 0, M*N*dtsize, (void*)hC.data());
  if(!is_correct(hC, rC, max_rounding_error(DTYPE(K))))
    exit(EXIT_FAILURE);
  stream.write(C, true, 0, M*N*dtsize, iC.data());

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, rs, rl, rgrid})){
    isaac::templates::GEMM gemm(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    //Compile
    std::string src;
    try{
      src = gemm.dump(ctx.device(), "gemm");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "gemm");

    //Launch
    gemm.enqueue(kernel, stream, sc_alpha, A, B, sc_beta, C);
    stream.synchronize();

    //Test
    stream.read(C, true, 0, M*N*dtsize, (void*)hC.data());
    stream.write(C, true, 0, M*N*dtsize, iC.data());
    size_t depth = x[11]*x[12]*x[13];
    double eps = max_rounding_error(DTYPE(K/depth))*depth;
    if(!is_correct(hC, rC, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix,
            size_t M, size_t N, size_t K,
            sc::IsaacOperation_t AT, sc::IsaacOperation_t BT)
{

  std::cout << "(" << M << ", " << N << ", " << K << ", " << AT << ", " << BT << ") [" << prefix << "]" << std::endl;
  do_test<DTYPE>(ctx, AT, BT, M, N, K);
  return EXIT_SUCCESS;
}

int main(){
  auto _N = sc::ISAAC_OP_N;
  auto _T = sc::ISAAC_OP_T;
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "GEMM:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "---------------" << std::endl;
  do_test<float>(ctx, "core, float", 67, 83, 673, _N, _N);
  do_test<float>(ctx, "core, float", 67, 83, 673, _N, _T);
  do_test<float>(ctx, "core, float", 67, 83, 673, _T, _N);
  do_test<float>(ctx, "core, float", 67, 83, 673, _T, _T);
  do_test<float>(ctx, "core, float", 1, 83, 673, _N, _N);
  do_test<float>(ctx, "core, float", 1, 83, 673, _N, _T);
  do_test<float>(ctx, "core, float", 1, 83, 673, _T, _N);
  do_test<float>(ctx, "core, float", 1, 83, 673, _T, _T);
  do_test<float>(ctx, "core, float", 67, 1, 673, _N, _N);
  do_test<float>(ctx, "core, float", 67, 1, 673, _N, _T);
  do_test<float>(ctx, "core, float", 67, 1, 673, _T, _N);
  do_test<float>(ctx, "core, float", 67, 1, 673, _T, _T);
  do_test<float>(ctx, "core, float", 67, 83, 1, _N, _N);
  do_test<float>(ctx, "core, float", 67, 83, 1, _N, _T);
  do_test<float>(ctx, "core, float", 67, 83, 1, _T, _N);
  do_test<float>(ctx, "core, float", 67, 83, 1, _T, _T);
  do_test<double>(ctx, "core, double", 67, 83, 673, _N, _N);
  do_test<double>(ctx, "core, double", 67, 83, 673, _N, _T);
  do_test<double>(ctx, "core, double", 67, 83, 673, _T, _N);
  do_test<double>(ctx, "core, double", 67, 83, 673, _T, _T);
  do_test<float>(ctx, "core + vectorized, float", 64, 96, 640, _N, _N);
  do_test<double>(ctx, "core + vectorized, double", 64, 96, 640, _N, _N);
  std::cout << "---------------" << std::endl;
}
