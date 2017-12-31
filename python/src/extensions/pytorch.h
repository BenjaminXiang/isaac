int isaac_conv_nd(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor *outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha, float scale,
                  THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);

int isaac_max_pool_nd(THCudaTensor *inputs, THCudaTensor *outputs,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      size_t stride_d, size_t stride_h, size_t stride_w);
