int isaac_conv_nd(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor *bias, THCudaTensor *outputs,
                  const char *activation, float alpha,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1);
