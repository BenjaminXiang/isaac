#include <THC/THC.h>
#include "isaac/api.h"

extern THCState *state;

extern "C"
{

inline isaac::ActivationType sc_activation(const std::string & activation){
    if(activation == "relu") return isaac::ReLU;
    if(activation == "linear") return isaac::Linear;
    if(activation == "sigmoid") return isaac::Sigmoid;
    throw std::runtime_error("Unknown activation function");
}


/* Convolution */
int isaac_conv_nd(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor *outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation, float alpha,
                  THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  int DIM = THCudaTensor_nDimension(state, inputs) - 2;

  // Datatype
  isaac::DType dtype = isaac::FLOAT_TYPE;

  // Inputs
  size_t N = THCudaTensor_size(state, inputs, 0);
  size_t Ci = THCudaTensor_size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = THCudaTensor_size(state, inputs, 2);
  if(DIM > 1) H = THCudaTensor_size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = THCudaTensor_size(state, inputs, 2 + (DIM > 2) + (DIM > 1));

  // Filter
  size_t T = 1, R = 1, S = 1;
  size_t Cf = THCudaTensor_size(state, filters, 0);
  if(DIM > 2) T = THCudaTensor_size(state, filters, 1);
  if(DIM > 1) R = THCudaTensor_size(state, filters, 1 + (DIM > 2));
  if(DIM > 0) S = THCudaTensor_size(state, filters, 1 + (DIM > 2) + (DIM > 1));
  long K = THCudaTensor_size(state, filters, 1 + DIM);

  if(Ci != Cf)
    return 1;
  size_t C = Ci;

  // Output shapes
  isaac::param_t M, P, Q;
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);

  // Create output
  size_t Zk = (z)?THCudaTensor_size(state, z, 1):0;
  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = K + Zk;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  THCudaTensor_resizeNd(state, outputs, 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, inputs)->data, false);
  isaac::driver::Buffer F(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, filters)->data, false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, outputs)->data, false);
  std::unique_ptr<isaac::driver::Buffer> Z;
  if(z)
    Z.reset(new isaac::driver::Buffer(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, z)->data, false));
  std::unique_ptr<isaac::driver::Buffer> Bias;
  if(bias)
    Bias.reset(new isaac::driver::Buffer(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, bias)->data, false));

  // Execute
  isaac::CONV(stream.context().device(), stream, dtype, dtype, N, K, M, P, Q, C, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              upsample_d, upsample_h, upsample_w,
              I, F, O,
              Bias.get(),
              sc_activation(activation), alpha,
              Zk, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, Z.get());

  return 1;
}

/* Max-pooling */
int isaac_max_pool_nd(THCudaTensor *inputs, THCudaTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t stride_d, size_t stride_h, size_t stride_w){
  int DIM = THCudaTensor_nDimension(state, inputs) - 2;

  // Datatype
  isaac::DType dtype = isaac::FLOAT_TYPE;

  // Inputs
  size_t N = THCudaTensor_size(state, inputs, 0);
  size_t C = THCudaTensor_size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = THCudaTensor_size(state, inputs, 2);
  if(DIM > 1) H = THCudaTensor_size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = THCudaTensor_size(state, inputs, 2 + (DIM > 2) + (DIM > 1));
  size_t T = window_d, R = window_h, S = window_w;

  // Output shapes
  isaac::param_t M, P, Q;
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 1, 1, 1, M, P, Q);

  // Create output
  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = C;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  THCudaTensor_resizeNd(state, outputs, 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, inputs)->data, false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, outputs)->data, false);

  // Execute
  isaac::POOL(stream.context().device(), stream, dtype, C, M, P, Q, N, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              I, O);

  return 1;
}

}
