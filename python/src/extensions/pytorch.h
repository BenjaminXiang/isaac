// Convolutions
int isaac_conv_nd_float_float(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                  THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);

int isaac_conv_nd_int_float(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                  THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);


int isaac_conv_nd_float_int(THCudaTensor *inputs, THCudaTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                  THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);

int isaac_conv_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  size_t quantized_in, size_t quantized_out, float iscale, float fscale, float* oscale, float zscale,
                  THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);

// Pooling
int isaac_max_pool_nd_float(THCudaTensor *inputs, THCudaTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t quantized,
                      size_t stride_d, size_t stride_h, size_t stride_w);

int isaac_max_pool_nd_int(THCudaIntTensor *inputs, THCudaIntTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t quantized,
                      size_t stride_d, size_t stride_h, size_t stride_w);


// Packing
int isaac_pack_nd(THCudaTensor* inputs, THCudaIntTensor* outputs, float a, float b);
