#include <THC/THC.h>
#include "isaac/api.h"

extern THCState *state;

extern "C"
{

inline isaac::ActivationType sc_activation(const std::string & activation){
    if(activation == "relu") return isaac::ReLU;
    if(activation == "linear") return isaac::Linear;
    throw std::runtime_error("Unknown activation function");
}

int isaac_conv_nd(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor *bias, THCudaTensor *outputs,
                  const char * activation, float alpha,
                  size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w)
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
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q);

  // Create output
  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = K;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  THCudaTensor_resizeNd(state, outputs, 2 + DIM, output_sizes, NULL);

  // Execute convolution
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, inputs)->data, false);
  isaac::driver::Buffer F(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, filters)->data, false);
  isaac::driver::Buffer Bias(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, bias)->data, false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, outputs)->data, false);
  isaac::CONV(stream.context().device(), stream, dtype, N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, I, F, O, &Bias, sc_activation(activation), alpha);

  return 1;
}

}
