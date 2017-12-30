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

template<class T> struct pack_increment{ enum{ VALUE = 1}; };
template<> struct pack_increment<int>{ enum{ VALUE = 4}; };

template <class T> T clamp(T x, T lo, T hi){
  return std::max(lo, std::min(x, hi));
}

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
    static const int PACK_OUT = pack_increment<DTYPE>::VALUE;
    Xc /= PACK_OUT;
    Yc /= PACK_OUT;
    size_t C = Xc + Yc;
    size_t Yd = Xd + crop_y_d0 + crop_y_d1;
    size_t Yh = Xh + crop_y_h0 + crop_y_h1;
    size_t Yw = Xw + crop_y_w0 + crop_y_w1;
    for(size_t n = 0; n < N; ++n)
    for(size_t xc = 0; xc < C ; ++xc)
    for(size_t xd = 0; xd < Xd; ++xd)
    for(size_t xh = 0; xh < Xh; ++xh)
    for(size_t xw = 0; xw < Xw; ++xw){
      size_t yd = xd + crop_y_d0;
      size_t yh = xh + crop_y_h0;
      size_t yw = xw + crop_y_w0;
      size_t idx_x = idx(n, xc, xd, xh, xw, N, Xc, Xd, Xh, Xw);
      size_t idx_y = idx(n, xc - Xc, yd, yh, yw, N, Yc, Yd, Yh, Yw);
      size_t idx_out = idx(n, xc, xd, xh, xw, N, C, Xd, Xh, Xw);
      out[idx_out] = (xc < Xc)?x[idx_x]:y[idx_y];
    }
}

float dot(float x, float y)
{ return x*y; }

inline int dot(int x, int y){
  int res = 0;
  for(int i = 0; i < 4; i++)
    res += ((x >> (8*i)) & 0x000000FF) * ((y >> (8*i)) & 0x000000FF);
  return res;
}

template<class T> T quantize_pack(float* tmp, float scale);
template<> float quantize_pack<float>(float* tmp, float scale){ return tmp[0]*scale; }
template<> int quantize_pack(float* tmp, float scale){
  int res = 0;
  for(int i = 0; i < 4; i++)
    res |= int8_t(clamp((int)(tmp[i]*scale), -128, 127)) << (8*i);
  return res;
}

template<class IN_DTYPE, class OUT_DTYPE>
void cpp_conv_nchw(int32_t C, int32_t N, int32_t K,
              int32_t D, int32_t H, int32_t W,
              int32_t T, int32_t R, int32_t S,
              int32_t pad_d, int32_t pad_h, int32_t pad_w,
              int32_t stride_d, int32_t stride_h, int32_t stride_w,
              int32_t M, int32_t P, int32_t Q,
              OUT_DTYPE* O, IN_DTYPE* I, IN_DTYPE* F,
              float* bias, float scale)
{
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  if(C % PACK_IN != 0) throw std::runtime_error("Number of input channels must be a multiple of 4");
  if(K % PACK_OUT != 0) throw std::runtime_error("Number of output channels must be a multiple of 4");
  C /= PACK_IN;
  K /= PACK_OUT;
  float tmp[PACK_OUT];
  for(int32_t m = 0 ; m < M; ++m)
  for(int32_t p = 0 ; p < P; ++p)
  for(int32_t q = 0; q < Q; ++q)
  for(int32_t n = 0; n < N; ++n)
  for(int32_t k = 0; k < K ; ++k)
  {
    for(int32_t i = 0; i < PACK_OUT; ++i)
      tmp[i] = 0;
    int32_t mm = m*stride_d - pad_d;
    int32_t pp = p*stride_h - pad_h;
    int32_t qq = q*stride_w - pad_w;
    for(int32_t kk = 0; kk < PACK_OUT; ++kk)
    for(int32_t c = 0; c < C; ++c)
    for(int32_t t = 0; t < T; ++t)
    for(int32_t r = 0; r < R; ++r)
    for(int32_t s = 0; s < S; ++s){
      int32_t d = mm + t;
      int32_t h = pp + r;
      int32_t w = qq + s;
      bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D && h < H && w < W);
      IN_DTYPE i = in_bounds?I[idx(n, c, d, h, w, N, C, D, H, W)]:0;
      IN_DTYPE f = F[idx(c, t, r, s, k*PACK_OUT + kk, C, T, R, S, K)];
      tmp[kk] += dot(i, f);
    }
    for(int32_t kk = 0; kk < PACK_OUT; ++kk)
      tmp[kk] += bias[k*PACK_OUT + kk];
    O[idx(n, k, m, p, q, N, K, M, P, Q)] = quantize_pack<OUT_DTYPE>(tmp, scale);
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
void do_test_impl(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  bool has_bias,
                  size_t Zk, size_t crop_z_m0, size_t crop_z_m1, size_t crop_z_p0, size_t crop_z_p1, size_t crop_z_q0, size_t crop_z_q1)
{
  srand(0);
  sc::DType in_dtype = sc::to_DType<IN_DTYPE>::value;
  sc::DType out_dtype = sc::to_DType<OUT_DTYPE>::value;

  size_t in_dtsize = sc::size_of(in_dtype);
  size_t out_dtsize = sc::size_of(out_dtype);

  sc::ActivationType activation = sc::Linear;
  drv::Stream stream(ctx);

  // Shapes
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);
  sc::param_t Dup = D*upsample_d, Hup = H*upsample_h, Wup = W*upsample_w;
  sc::param_t Zm = M + crop_z_m0 + crop_z_m1;
  sc::param_t Zp = P + crop_z_p0 + crop_z_p1;
  sc::param_t Zq = Q + crop_z_q0 + crop_z_q1;

  // CPU buffers
  size_t PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  size_t PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  std::vector<IN_DTYPE> image_c(N*C*H*W*D/PACK_IN);
  std::vector<IN_DTYPE> upsampled_c(N*C*Hup*Wup*Dup/PACK_IN);
  std::vector<IN_DTYPE> filters_c(K*C*R*S*T/PACK_IN);
  std::vector<float> bias_c(K);
  std::vector<OUT_DTYPE> conv_c(N*K*M*P*Q/PACK_OUT);
  std::vector<OUT_DTYPE> z_c(N*Zk*Zm*Zp*Zq/PACK_OUT);
  std::vector<OUT_DTYPE> ground_truth_c(N*(K + Zk)*M*P*Q/PACK_OUT);
  std::vector<OUT_DTYPE> output_isaac_c(ground_truth_c);
  // Initialize
  for(size_t i = 0; i < z_c.size(); ++i)
    z_c[i] = (float)rand()/RAND_MAX*10;
  for(size_t i = 0; i < image_c.size(); ++i)
    image_c[i] = (float)rand()/RAND_MAX*10;
  for(size_t i = 0; i < filters_c.size(); ++i)
    filters_c[i] = (float)rand()/RAND_MAX*10;
  for(size_t i = 0; i < bias_c.size(); ++i)
    bias_c[i] = has_bias?(float)rand()/RAND_MAX*10:0;
  float scale = (out_dtype==sc::INT8X4_TYPE)?1./(C*R*S*T)*5.:1.;

  // Ground truth
  upsample(image_c, upsampled_c, N, C/PACK_IN, D, H, W, upsample_d, upsample_h, upsample_w);
  cpp_conv_nchw(C, N, K, Dup, Hup, Wup, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q, conv_c.data(), upsampled_c.data(), filters_c.data(), bias_c.data(), scale);
  crop_merge(conv_c, z_c, ground_truth_c, N, K, M, P, Q, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1); //crop_merge

  // Isaac
  drv::Buffer image(ctx, image_c.size()*in_dtsize);
  drv::Buffer filters(ctx, filters_c.size()*in_dtsize);
  drv::Buffer output(ctx, ground_truth_c.size()*out_dtsize);
  drv::Buffer z(ctx, std::max<int>(1, z_c.size()*out_dtsize));
  drv::Buffer bias(ctx, std::max<int>(1, bias_c.size()*out_dtsize));
  drv::Buffer* pz = Zk>0?&z:NULL;
  drv::Buffer* pbias = has_bias?&bias:NULL;
  stream.write(image, false, 0, image_c);
  stream.write(filters, false, 0, filters_c);
  stream.write(z, false, 0, z_c);
  stream.write(bias, false, 0, bias_c);
  sc::CONV(ctx.device(), stream, in_dtype, out_dtype, N, K, M, P, Q, C, T, R, S, D, H, W,
           pad_d, pad_h, pad_w,
           stride_d, stride_h, stride_w,
           upsample_d, upsample_h, upsample_w,
           image, filters, output,
           pbias,
           activation, 0,
           scale, Zk,
           crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1, pz);
  stream.read(output, true, 0, output_isaac_c);

  // Check correctness
  if(!is_correct(output_isaac_c, ground_truth_c, max_rounding_error(float(C))))
    exit(EXIT_FAILURE);

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rs, rs, rl, r1, rgrid, rgrid})){
    isaac::templates::Conv conv(in_dtype, out_dtype, C, D, H, W, N, K, M, P, Q, T, R, S,
                                pad_d, pad_h, pad_w,
                                stride_d, stride_h, stride_w,
                                upsample_d, upsample_h, upsample_w,
                                activation,
                                Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1,
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
      conv.enqueue(kernel, stream, image, filters, output, pbias, 0, 1., pz);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    //Test
    stream.read(output, true, 0, output_isaac_c.size()*out_dtsize, (void*)output_isaac_c.data());
    size_t depth = x[6]*x[7]*x[8];
    double eps = max_rounding_error(float(C/depth))*depth;
    if(!is_correct(output_isaac_c, ground_truth_c, eps))
      exit(EXIT_FAILURE);
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
            size_t pad_d, size_t pad_h, size_t pad_w,
            size_t stride_d, size_t stride_h, size_t stride_w,
            size_t upsample_d, size_t upsample_h, size_t upsample_w,
            bool has_bias,
            size_t Zk, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  auto params = {N, K, D, H, W, C, T, R, S};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b) [" << prefix << "]" << std::endl;
  do_test_impl<IN_DTYPE, OUT_DTYPE>(ctx, N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, has_bias, Zk, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << " FLOAT x INT:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "CONV: FPROP" << std::endl;
  std::cout << "-----------" << std::endl;
  do_test<int, float>(ctx, "core", 5, 16, 19, 11, 15, 20, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 0, 0, 0, 0, 0, 0, 0);

//  do_test<float, float>(ctx, "core", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "upsample", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 3, 2, 4, false, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "crop-merge", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 77, 1, 3, 5, 4, 2, 6);
//  do_test<float, float>(ctx, "pad", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 1, 1, 1, 1, 1, 1, false, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "stride", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 6, 3, 4, 1, 1, 1, false, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "pad + stride + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "vectorized + bias", 5, 13, 36, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, true, 0, 0, 0, 0, 0, 0, 0);
//  do_test<float, float>(ctx, "pad + stride + crop-merge + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 77, 1, 3, 5, 4, 2, 6);
//  do_test<float, float>(ctx, "upsample + crop-merge + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, true, 77, 1, 3, 5, 4, 2, 6);
//  do_test<float, float>(ctx, "pad + stride + crop-merge + bias", 5, 13, 19, 11, 15, 17, 1, 1, 1, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 77, 1, 3, 5, 4, 2, 6);
//  do_test<float, float>(ctx, "upsample + crop-merge + bias", 5, 13, 19, 11, 15, 17, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, true, 77, 1, 3, 5, 4, 2, 6);
  std::cout << "-----------" << std::endl;
}
