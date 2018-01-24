#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <iterator>
#include <cassert>
#include <cmath>
#include <cfenv>


#include "isaac/driver/backend.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/conv.h"
#include "isaac/templates/error.hpp"

#include "isaac/driver/cublas.h"

#include "isaac/tools/collections.hpp"
#include "isaac/api.h"

#include "test_utils.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

inline int32_t idx(int32_t x, int32_t y, int32_t z, int32_t w, int32_t u,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3, int32_t s4)
{ return u + w*s4 + z*s4*s3 + y*s4*s3*s2 + x*s4*s3*s2*s1; }

template<class DTYPE>
inline void to_cudnn(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t C, size_t D, size_t H, size_t W, size_t N){
  for(size_t c = 0; c < C ; ++c)
  for(size_t d = 0; d < D; ++d)
  for(size_t h = 0; h < H; ++h)
  for(size_t w = 0; w < W; ++w)
  for(size_t n = 0; n < N; ++n)
    out[idx(n, c, d, h, w, N, C, D, H, W)] = in[idx(c, d, h, w, n, C, D, H, W, N)];
}

template<class DTYPE>
inline void from_cudnn(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t N, size_t K, size_t M, size_t P, size_t Q){
    for(size_t k = 0; k < K ; ++k)
    for(size_t m = 0; m < M; ++m)
    for(size_t p = 0; p < P; ++p)
    for(size_t q = 0; q < Q; ++q)
    for(size_t n = 0; n < N; ++n)
      out[idx(k, m, p, q, n, K, M, P, Q, N)] = in[idx(n, k, m, p, q, N, K, M, P, Q)];
}

template <class T> T clamp(T x, T lo, T hi){
  return std::max<T>(lo, std::min<T>(x, hi));
}

float max(float x, float y)
{
  return std::max(x, y);
}

float add(float x, float y)
{
  return x + y;
}

float* unpack(float* ptr, float value, float scale)
{
  *ptr = value/scale;
  return ptr;
}

float* unpack(float* ptr, int value, float scale)
{
  for(int i = 0; i < 4; i++){
    int shifted = (value >> (8*i) & 0xff);
    ptr[i] = ((float)(*(int8_t*)(&shifted)))/scale;
  }
  return ptr;
}

template<class T> T pack(float* tmp, float scale);

template<> float pack<float>(float* tmp, float scale)
{ return tmp[0]*scale; }

template<> int pack(float* tmp, float scale)
{
  int res = 0;
  for(int i = 0; i < 4; i++){
    int8_t clamped = std::round(clamp(tmp[i]*scale, (float)-128, (float)127));
    res |= (clamped & 0xFF) << (8*i);
  }
  return res;
}


template<class T> struct pack_increment{ enum{ VALUE = 1}; };
template<> struct pack_increment<int32_t>{ enum{ VALUE = 4}; };

template<class T> struct init_acc;
template<> struct init_acc<float>{ static constexpr float value = -INFINITY; };
template<> struct init_acc<int32_t>{ static constexpr int32_t value = 0x80808080; };

template<class DTYPE>
void cpp_pool(sc::PoolType pool_type,
              int32_t N, int32_t C,
              int32_t D, int32_t H, int32_t W,
              int32_t T, int32_t R, int32_t S,
              int32_t pad_d, int32_t pad_h, int32_t pad_w,
              int32_t stride_d, int32_t stride_h, int32_t stride_w,
              int32_t M, int32_t P, int32_t Q,
              DTYPE* O, DTYPE* I)
{
  static const int PACK = pack_increment<DTYPE>::VALUE;
  if(C % PACK != 0) throw std::runtime_error("Number of input channels must be a multiple of 4");
  C /= PACK;
  float TRS = T*R*S;
  std::function<DTYPE(DTYPE, DTYPE)> accumulate = (pool_type == sc::MaxPool)?&max:&add;

  float acc[PACK];
  float unpacked[PACK];

  for(int32_t m = 0 ; m < M; ++m)
  for(int32_t p = 0 ; p < P; ++p)
  for(int32_t q = 0; q < Q; ++q)
  for(int32_t n = 0; n < N; ++n)
  for(int32_t c = 0; c < C ; ++c)
  {
    int32_t mm = m*stride_d - pad_d;
    int32_t pp = p*stride_h - pad_h;
    int32_t qq = q*stride_w - pad_w;

    // Initialize accumulators
    for(size_t j = 0; j < PACK ; j++)
      acc[j] = (sc::MaxPool)?-INFINITY:0;

    // Accumulate
    for(int32_t t = 0; t < T; ++t)
    for(int32_t r = 0; r < R; ++r)
    for(int32_t s = 0; s < S; ++s){
      int32_t d = mm + t;
      int32_t h = pp + r;
      int32_t w = qq + s;
      bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D && h < H && w < W);
      DTYPE i = in_bounds?I[idx(n, c, d, h, w, N, C, D, H, W)]:0;
      unpack(unpacked, i, 1.);
      for(size_t j = 0; j < PACK ; j++)
        acc[j] = accumulate(acc[j], unpacked[j]);
    }

    // Write back
    O[idx(n, c, m, p, q, N, C, M, P, Q)] = pack<DTYPE>(acc, (pool_type==sc::AvgPool)?1./TRS:1.);
  }

}

template<class DTYPE>
void do_test_impl(sc::driver::Context const & ctx, isaac::PoolType pool_type, size_t N, size_t K, size_t D, size_t H, size_t W, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w){
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);
  static const int PACK = pack_increment<DTYPE>::VALUE;

  //Initialize input/output buffers
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 1, 1, 1, M, P, Q);

  std::vector<DTYPE> iO(N*K*P*Q*M/PACK);
  std::vector<DTYPE> iI(N*K*H*W*D/PACK);
  std::vector<DTYPE> rO(iO.size());
  drv::Buffer O(ctx, iO.size()*dtsize);
  drv::Buffer I(ctx, iI.size()*dtsize);
  srand(0);
  for(size_t i = 0; i < iI.size(); ++i) iI[i] = (dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX;

  //Ground result (cuDNN)
  cpp_pool(pool_type, N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q, rO.data(), iI.data());

  //Test ISAAC
  drv::Stream stream(ctx);
  stream.write(I, true, 0, iI.size()*dtsize, iI.data());
  sc::POOL(ctx.device(), stream, dtype, pool_type, K, M, P, Q, N, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, I, O);
  stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
  if(!is_correct(iO, rO, max_rounding_error(DTYPE(T*R*S))))
    exit(EXIT_FAILURE);

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {32, 64, 128, 256};
  std::vector<int> rs = {4, 8};
  for(auto x: sc::cpp::cartesian({rv, rl, rs, std::vector<int>{4}})){
    isaac::templates::Pool pool(dtype, pool_type, K, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                                x[0], x[1], x[2], x[3]);
    //Compile
    std::string src;
    try{
      src = pool.dump(ctx.device(), "pool");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    //Compile
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "pool");
    //Launch
    try{
      pool.enqueue(kernel, stream, I, O);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    stream.synchronize();
    //Test
    stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
    double eps = max_rounding_error(DTYPE(T*R*S));
    if(!is_correct(iO, rO, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix, sc::PoolType pool_type, size_t N, size_t K, size_t D, size_t H, size_t W, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w){
  auto params = {N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b) [" << prefix << "]" << std::endl;
  do_test_impl<DTYPE>(ctx, pool_type, N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "POOL" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "---------------" << std::endl;
  do_test<float>(ctx, "max + core", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<int32_t>(ctx, "max + int8x4", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<float>(ctx, "avg + core", sc::AvgPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<int32_t>(ctx, "avg + int8x4", sc::AvgPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<float>(ctx, "max + stride", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 6, 3, 4);
  do_test<float>(ctx, "max + pad", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 1, 2, 1, 1, 1, 1);
  do_test<float>(ctx, "max + pad + stride", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 1, 2, 1, 6, 3, 4);
  do_test<int32_t>(ctx, "max + int8x4 + stride", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 6, 3, 4);
  do_test<int32_t>(ctx, "max + int8x4 + pad", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 1, 2, 1, 1, 1, 1);
  do_test<int32_t>(ctx, "max + int8x4 + pad + stride", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 1, 2, 1, 6, 3, 4);
  std::cout << "---------------" << std::endl;
}
