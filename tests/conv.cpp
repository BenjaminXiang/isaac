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
inline void crop_merge(std::vector<DTYPE> const & x, std::vector<DTYPE> const & y, std::vector<DTYPE>& out,
                     size_t N, size_t Xc, size_t Xd, size_t Xh, size_t Xw,
                     size_t Yc, size_t crop_y_d0, size_t crop_y_d1, size_t crop_y_h0,size_t crop_y_h1, size_t crop_y_w0, size_t crop_y_w1)
{
    size_t C = Xc + Yc;
    size_t Yd = Xd + crop_y_d0 + crop_y_d1;
    size_t Yh = Xh + crop_y_h0 + crop_y_h1;
    size_t Yw = Xw + crop_y_w0 + crop_y_w1;

    std::cout << Xc << std::endl;
    for(size_t n = 0; n < N; ++n)
    for(size_t xc = 0; xc < C ; ++xc)
    for(size_t xd = 0; xd < Xd; ++xd)
    for(size_t xh = 0; xh < Xh; ++xh)
    for(size_t xw = 0; xw < Xw; ++xw){
      size_t yd = xd + crop_y_d0;
      size_t yh = xh + crop_y_h0;
      size_t yw = xh + crop_y_w0;
      size_t idx_x = idx(n, xc, xd, xh, xw, N, Xc, Xd, Xh, Xw);
      size_t idx_y = idx(n, xc - Xc, yd, yh, yw, N, Yc, Yd, Yh, Yw);
      size_t idx_out = idx(n, xc, xd, xh, xw, N, C, Xd, Xh, Xw);
      out[idx_out] = (xc < Xc)?x[idx_x]:0;
      if(idx_out == 417339) std::cout << xc  << " " << Xc << " " << xc - Xc << " "  << idx_y << " " << y[idx_y] << " " << out[idx_out] << std::endl;
    }
}


template<class DTYPE>
void do_test_impl(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t Zk, size_t crop_z_m0, size_t crop_z_m1, size_t crop_z_p0, size_t crop_z_p1, size_t crop_z_q0, size_t crop_z_q1)
{
  srand(0);
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);
  sc::ActivationType activation = sc::Linear;
  drv::Stream stream(ctx);

  //alpha, beta are not half-precision
  sc::DType ab_dtype = (dtype==sc::INT8X4_TYPE)?sc::FLOAT_TYPE:dtype;
  sc::scalar alpha(1., ab_dtype), beta(0., ab_dtype);

  // Shapes
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);
  sc::param_t Dup = D*upsample_d, Hup = H*upsample_h, Wup = W*upsample_w;
  sc::param_t Zm = M + crop_z_m0 + crop_z_m1;
  sc::param_t Zp = P + crop_z_p0 + crop_z_p1;
  sc::param_t Zq = Q + crop_z_q0 + crop_z_q1;

  // CPU buffers
  size_t vect_c = (dtype==sc::INT8X4_TYPE)?4:1;
  std::vector<DTYPE> image_cpu(N*C/vect_c*H*W*D);
  std::vector<DTYPE> upsampled_cpu(N*C/vect_c*Hup*Wup*Dup);
  std::vector<DTYPE> filters_cpu(K*C/vect_c*R*S*T);
  std::vector<DTYPE> conv_cpu(N*K*M*P*Q);
  std::vector<DTYPE> z_cpu(N*Zk*Zm*Zp*Zq);
  std::vector<DTYPE> output_cpu(N*(K + Zk)*M*P*Q);
  std::vector<DTYPE> output_cpu_isaac(output_cpu);

  std::vector<DTYPE> filters_cudnn_cpu(filters_cpu.size());
  for(size_t i = 0; i < z_cpu.size(); ++i) z_cpu[i] = 0;
  for(size_t i = 0; i < image_cpu.size(); ++i) image_cpu[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < filters_cpu.size(); ++i) filters_cpu[i] = (float)rand()/RAND_MAX;
  to_cudnn(filters_cpu, filters_cudnn_cpu, C/vect_c, T, R, S, K);

  // GPU buffers
  drv::Buffer image(ctx, image_cpu.size()*dtsize);
  drv::Buffer upsampled(ctx, upsampled_cpu.size()*dtsize);
  drv::Buffer filters(ctx, filters_cpu.size()*dtsize);
  drv::Buffer conv(ctx, conv_cpu.size()*dtsize);
  drv::Buffer z(ctx, z_cpu.size()*dtsize);
  drv::Buffer output(ctx, output_cpu.size()*dtsize);

  // Ground truth
  upsample(image_cpu, upsampled_cpu, N, C, D, H, W, upsample_d, upsample_h, upsample_w);
  stream.write(upsampled, true, 0, upsampled_cpu.size()*dtsize, upsampled_cpu.data()); // upsample
  stream.write(filters, true, 0, filters_cpu.size()*dtsize, filters_cudnn_cpu.data());
  sc::driver::cudnnConv(dtype, stream, Dup, Hup, Wup, N, K, M, P, Q, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, upsampled, filters, beta, conv); // conv
  stream.read(conv, true, 0, conv_cpu.size()*dtsize, (void*)conv_cpu.data());
  std::cout << idx(0,0,29,27,13,N,Zk,Zm,Zp,Zq) << " " << z_cpu[idx(0,0,29,27,13,N,Zk,Zm,Zp,Zq)] << std::endl;
  crop_merge(conv_cpu, z_cpu, output_cpu, N, K, M, P, Q, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1); //crop_merge

  // Isaac
  stream.write(image, true, 0, image_cpu.size()*dtsize, image_cpu.data());
  stream.write(filters, true, 0, filters_cpu.size()*dtsize, filters_cpu.data());
  stream.write(z, true, 0, z_cpu.size()*dtsize, z_cpu.data());
  sc::CONV(ctx.device(), stream, dtype, N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, image, filters, output, NULL, activation, 0, Zk);
  stream.read(output, true, 0, output_cpu_isaac.size()*dtsize, (void*)output_cpu_isaac.data());
  if(!is_correct(output_cpu_isaac, output_cpu, max_rounding_error(DTYPE(C))))
    exit(EXIT_FAILURE);
  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rs, rs, rl, r1, rgrid, rgrid})){
    isaac::templates::Conv conv(dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, activation, false,
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
      conv.enqueue(kernel, stream, image, filters, output, NULL, 0, Zk);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    //Test
    stream.read(output, true, 0, output_cpu_isaac.size()*dtsize, (void*)output_cpu_isaac.data());
    size_t depth = x[6]*x[7]*x[8];
    double eps = max_rounding_error(DTYPE(C/depth))*depth;
    if(!is_correct(output_cpu_isaac, output_cpu, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
            size_t pad_d, size_t pad_h, size_t pad_w,
            size_t stride_d, size_t stride_h, size_t stride_w,
            size_t upsample_d, size_t upsample_h, size_t upsample_w,
            size_t Zk, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  auto params = {N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b)" << std::endl;
  do_test_impl<DTYPE>(ctx, N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, Zk, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "FLOAT:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "CONV: FPROP" << std::endl;
  std::cout << "-----------" << std::endl;
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, 23, 0, 0, 0, 0, 0, 0);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 5, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 0, 0, 0, 6, 3, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
  do_test<float>(ctx, 5, 41, 31, 29, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
  std::cout << "-----------" << std::endl;
}
