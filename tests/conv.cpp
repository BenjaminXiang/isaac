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
#include <iterator>

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


template<class DTYPE>
inline void upsample(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t N, size_t C, size_t D, size_t H, size_t W, size_t upsample_d, size_t upsample_h, size_t upsample_w){
    for(size_t n = 0; n < N; ++n)
    for(size_t c = 0; c < C ; ++c)
    for(size_t d = 0; d < D; ++d)
    for(size_t h = 0; h < H; ++h)
    for(size_t w = 0; w < W; ++w){
    for(size_t ud = 0; ud < upsample_d; ud++)
    for(size_t uh = 0; uh < upsample_h; uh++)
    for(size_t uw = 0; uw < upsample_w; uw++)
      out[idx(n, c, d*upsample_d + ud, h*upsample_h + uh, w*upsample_w + uw, N, C, D*upsample_d, H*upsample_h, W*upsample_w)] = in[idx(n, c, d, h, w, N, C, D, H, W)];
    }
}


template<class DTYPE>
void do_test_impl(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w){
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);
  sc::ActivationType activation = sc::Linear;

  //alpha, beta are not half-precision
  sc::DType ab_dtype = (dtype==sc::INT8X4_TYPE)?sc::FLOAT_TYPE:dtype;
  sc::scalar alpha(1., ab_dtype), beta(0., ab_dtype);

  //Initialize input/output buffers
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);
  sc::param_t Dup = D*upsample_d, Hup = H*upsample_h, Wup = W*upsample_w;

  size_t vect_c = (dtype==sc::INT8X4_TYPE)?4:1;
  std::vector<DTYPE> iO(N*K*P*Q*M);
  std::vector<DTYPE> iI(N*C/vect_c*H*W*D);
  std::vector<DTYPE> iI_upsampled(N*C/vect_c*Hup*Wup*Dup);
  std::vector<DTYPE> iF(K*C/vect_c*R*S*T);
  drv::Buffer O(ctx, iO.size()*dtsize);
  drv::Buffer I(ctx, iI.size()*dtsize);
  drv::Buffer I_upsampled(ctx, iI_upsampled.size()*dtsize);
  drv::Buffer F(ctx, iF.size()*dtsize);
  srand(0);
  for(size_t i = 0; i < iI.size(); ++i) iI[i] = i;
  for(size_t i = 0; i < iF.size(); ++i) iF[i] = 1;
  std::vector<DTYPE> iF_cudnn(iF.size());
  to_cudnn(iF, iF_cudnn, C/vect_c, T, R, S, K);
  upsample(iI, iI_upsampled, N, C, D, H, W, upsample_d, upsample_h, upsample_w);

//  std::cout << iI_upsampled[idx(0, 0, 0, 0, 0, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 0, 1, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 0, 2, N, C, Dup, Hup, Wup)] << std::endl;
//  std::cout << iI_upsampled[idx(0, 0, 0, 1, 0, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 1, 1, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 1, 2, N, C, Dup, Hup, Wup)] << std::endl;
//  std::cout << iI_upsampled[idx(0, 0, 0, 2, 0, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 2, 1, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 2, 2, N, C, Dup, Hup, Wup)] << std::endl;
//  std::cout << iI_upsampled[idx(0, 0, 0, 3, 0, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 3, 1, N, C, Dup, Hup, Wup)] << " " << iI_upsampled[idx(0, 0, 0, 3, 2, N, C, Dup, Hup, Wup)] << std::endl;



  //Ground result (cuDNN)
  drv::Stream stream(ctx);
  stream.write(O, true, 0, iO.size()*dtsize, iO.data());
  stream.write(I_upsampled, true, 0, iI_upsampled.size()*dtsize, iI_upsampled.data());
  stream.write(F, true, 0, iF.size()*dtsize, iF_cudnn.data());
  sc::driver::cudnnConv(dtype, stream, Dup, Hup, Wup, N, K, M, P, Q, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, I_upsampled, F, beta, O);
  std::vector<DTYPE> rO(iO.size());
  stream.read(O, true, 0, rO.size()*dtsize, (void*)rO.data());
  stream.write(O, true, 0, iO.size()*dtsize, iO.data());
  stream.write(I, true, 0, iI.size()*dtsize, iI.data());
  stream.write(F, true, 0, iF.size()*dtsize, iF.data());

  //Test ISAAC
  sc::CONV(ctx.device(), stream, dtype, N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, I, F, O, NULL, activation);
  stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
  if(!is_correct(iO, rO, max_rounding_error(DTYPE(C))))
    exit(EXIT_FAILURE);

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rs, rs, rl, r1, rgrid, rgrid})){
    isaac::templates::Conv conv(dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, activation,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
    //Compile
    std::string src;
    try{
      src = conv.dump(ctx.device(), "fprop");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    //Compile
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "fprop");
    //Launch
    try{
      conv.enqueue(kernel, stream, I, F, O);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    stream.synchronize();
    //Test
    stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
    size_t depth = x[6]*x[7]*x[8];
    double eps = max_rounding_error(DTYPE(C/depth))*depth;
    if(!is_correct(iO, rO, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w, size_t upsample_d, size_t upsample_h, size_t upsample_w){
  auto params = {N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b)" << std::endl;
  do_test_impl<DTYPE>(ctx, N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "FLOAT:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "CONV: FPROP" << std::endl;
  std::cout << "-----------" << std::endl;
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 2, 2);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 5, 1, 2, 1, 1, 1, 1, 1, 1);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 0, 0, 0, 6, 3, 4, 1, 1, 1);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1);
  std::cout << "-----------" << std::endl;
}
