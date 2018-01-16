#include <THC/THC.h>
#include "isaac/api.h"

extern THCState *state;

#define WRAP0(RET, NAME, TYPE0) RET NAME(THCState *state, TYPE0 *x){ return TYPE0 ## _ ## NAME(state,x);}
#define WRAP1(RET, NAME, TYPE0, TYPE1) RET NAME(THCState *state, TYPE0 *x, TYPE1 arg0){ return TYPE0 ## _ ## NAME(state, x, arg0);}

WRAP0(int, nDimension, THCudaTensor)
WRAP0(int, nDimension, THCudaIntTensor)
WRAP1(int, size, THCudaTensor, int)
WRAP1(int, size, THCudaIntTensor, int)
WRAP0(THCudaStorage*, storage, THCudaTensor)
WRAP0(THCudaIntStorage*, storage, THCudaIntTensor)
void resizeNd(THCState *state, THCudaTensor *tensor, int nDimension, long *size, long *stride)
{ return THCudaTensor_resizeNd(state, tensor, nDimension, size, stride);}
void resizeNd(THCState *state, THCudaIntTensor *tensor, int nDimension, long *size, long *stride)
{ return THCudaIntTensor_resizeNd(state, tensor, nDimension, size, stride);}


inline isaac::ActivationType get_sc_activation(const std::string & activation){
    if(activation == "relu") return isaac::ReLU;
    if(activation == "linear") return isaac::Linear;
    if(activation == "sigmoid") return isaac::Sigmoid;
    throw std::runtime_error("Unknown activation function");
}

inline isaac::ResidualType get_sc_residual(const std::string & residual){
    if(residual == "") return isaac::NoResidual;
    if(residual == "cat") return isaac::CatResidual;
    if(residual == "add") return isaac::AddResidual;
    throw std::runtime_error("Unknown residual function");
}

/* Convolution */
template<typename IN_TYPE, typename OUT_TYPE>
int isaac_conv_nd_impl(IN_TYPE *inputs, IN_TYPE *filters, OUT_TYPE **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation, float alpha,
                  size_t quantized_in, size_t quantized_out, float iscale, float fscale, float * oscale, float zscale,
                  const char * residual, OUT_TYPE *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  int DIM = nDimension(state, inputs) - 2;
  isaac::ActivationType sc_activation = get_sc_activation(activation);
  isaac::ResidualType sc_residual = get_sc_residual(residual);

  // Datatype
  isaac::DType in_dtype = quantized_in?isaac::INT8X4_TYPE:isaac::FLOAT_TYPE;
  isaac::DType out_dtype = quantized_out?isaac::INT8X4_TYPE:isaac::FLOAT_TYPE;
  size_t vect_c = quantized_in?4:1;
  size_t vect_k = quantized_out?4:1;

  // Inputs
  size_t N = size(state, inputs, 0);
  size_t Ci = size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = size(state, inputs, 2);
  if(DIM > 1) H = size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = size(state, inputs, 2 + (DIM > 2) + (DIM > 1));

  // Filter
  size_t T = 1, R = 1, S = 1;
  size_t Cf = size(state, filters, 0);
  if(DIM > 2) T = size(state, filters, 1);
  if(DIM > 1) R = size(state, filters, 1 + (DIM > 2));
  if(DIM > 0) S = size(state, filters, 1 + (DIM > 2) + (DIM > 1));
  long K = size(state, filters, 1 + DIM);

  if(Ci != Cf)
    return 0;
  size_t C = Ci;

  // Output shapes
  isaac::param_t M, P, Q;
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);

  // Create output
  size_t Zk = (z)?size(state, z, 1):0;
  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = K/vect_k;
  if(sc_residual == isaac::CatResidual)
    output_sizes[1] += Zk;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  for(int i = 0; i < num_outputs; i++)
    resizeNd(state, outputs[i], 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)storage(state, inputs)->data, false);
  isaac::driver::Buffer F(stream.context(), (CUdeviceptr)storage(state, filters)->data, false);
  std::vector<isaac::driver::Buffer> O;
  for(int i = 0; i < num_outputs; i++)
    O.push_back(isaac::driver::Buffer(stream.context(), (CUdeviceptr)storage(state, outputs[i])->data, false));
  std::unique_ptr<isaac::driver::Buffer> Z;
  if(z)
    Z.reset(new isaac::driver::Buffer(stream.context(), (CUdeviceptr)storage(state, z)->data, false));
  std::unique_ptr<isaac::driver::Buffer> Bias;
  if(bias)
    Bias.reset(new isaac::driver::Buffer(stream.context(), (CUdeviceptr)storage(state, bias)->data, false));

  // Execute
  isaac::CONV(stream.context().device(), stream, in_dtype, out_dtype, N, K, M, P, Q, C*vect_c, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              upsample_d, upsample_h, upsample_w,
              I, F, O.data(), num_outputs,
              Bias.get(),
              sc_activation, alpha,
              iscale, fscale, std::vector<float>(oscale, oscale + num_outputs), zscale,
              sc_residual, Zk*vect_k, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, Z.get());

  return 1;
}

template<class TYPE>
int isaac_max_pool_nd_impl(TYPE *inputs, TYPE *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t quantized,
                      size_t stride_d, size_t stride_h, size_t stride_w){
  int DIM = nDimension(state, inputs) - 2;

  // Datatype
  isaac::DType dtype = quantized?isaac::INT8X4_TYPE:isaac::FLOAT_TYPE;
  size_t vect_c = quantized?4:1;

  // Inputs
  size_t N = size(state, inputs, 0);
  size_t C = size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = size(state, inputs, 2);
  if(DIM > 1) H = size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = size(state, inputs, 2 + (DIM > 2) + (DIM > 1));
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
  resizeNd(state, outputs, 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)storage(state, inputs)->data, false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)storage(state, outputs)->data, false);

  // Execute
  isaac::POOL(stream.context().device(), stream, dtype, C*vect_c, M, P, Q, N, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              I, O);

  return 1;
}

extern "C"
{


int isaac_conv_nd_float_float(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  return isaac_conv_nd_impl(inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            quantized_in, quantized_out, iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
}

int isaac_conv_nd_int_float(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  return isaac_conv_nd_impl(inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            quantized_in, quantized_out, iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
}

int isaac_conv_nd_float_int(THCudaTensor *inputs, THCudaTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  return isaac_conv_nd_impl(inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            quantized_in, quantized_out, iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
}

int isaac_conv_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  return isaac_conv_nd_impl(inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            quantized_in, quantized_out, iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
}

/* Max-pooling */
int isaac_max_pool_nd_float(THCudaTensor *inputs, THCudaTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t quantized,
                      size_t stride_d, size_t stride_h, size_t stride_w){
  return isaac_max_pool_nd_impl(inputs, outputs, window_d, window_h, window_w, pad_d, pad_h, pad_w, quantized, stride_d, stride_h, stride_w);
}


int isaac_max_pool_nd_int(THCudaIntTensor *inputs, THCudaIntTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t quantized,
                      size_t stride_d, size_t stride_h, size_t stride_w){
  return isaac_max_pool_nd_impl(inputs, outputs, window_d, window_h, window_w, pad_d, pad_h, pad_w, quantized, stride_d, stride_h, stride_w);
}


/* Transform */
int isaac_pack_nd(THCudaTensor* inputs, THCudaIntTensor *outputs, float a, float b){
  size_t DIM = THCudaTensor_nDimension(state, inputs);
  std::vector<long> sizes(DIM);
  std::memcpy(sizes.data(), inputs->size, DIM*sizeof(long));

  // Allocate output
  if(sizes[1] % 4 != 0)
    return 0;
  sizes[1] /= 4;
  THCudaIntTensor_resizeNd(state, outputs, DIM, sizes.data(), NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)THCudaTensor_storage(state, inputs)->data, false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)THCudaIntTensor_storage(state, outputs)->data, false);
  isaac::scalar alpha(a, isaac::FLOAT_TYPE);
  isaac::scalar beta(b, isaac::FLOAT_TYPE);

  // Execute
  long D = (DIM > 4)?sizes[2]:1;
  long H = (DIM > 3)?sizes[2 + (DIM>4)]:1;
  long W = (DIM > 2)?sizes[2 + (DIM>4) + (DIM>3)]:1;
  isaac::driver::cudnnTransformTensor(stream, isaac::FLOAT_TYPE, isaac::INT8X4_TYPE, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW_VECT_C,
                                      sizes[0], sizes[1]*4, D, H, W, alpha, I, beta, O);

  return 1;
}

}
