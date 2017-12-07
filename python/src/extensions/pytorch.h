int isaac_conv_nd(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor *bias, THCudaTensor *outputs, const char *activation, float alpha,
                  size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w);
