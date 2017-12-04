/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_SCALAR_H
#define ISAAC_SCALAR_H

#include "isaac/external/half.hpp"

namespace isaac{


enum DType{
  INT8X4_TYPE = 1,
  INT32_TYPE,
  FLOAT_TYPE,
  DOUBLE_TYPE,
};

inline size_t size_of(DType dtype){
  switch (dtype) {
  case INT8X4_TYPE: return 4;
  case INT32_TYPE: return 4;
  case FLOAT_TYPE: return 4;
  case DOUBLE_TYPE: return 8;
  default: throw;
  }
}

template<class T> struct to_DType;
template<> struct to_DType<int32_t>{ static const DType value = INT8X4_TYPE; };
template<> struct to_DType<float>{ static const DType value = FLOAT_TYPE; };
template<> struct to_DType<double>{ static const DType value = DOUBLE_TYPE; };

class scalar{
private:
  template<class T>
  void init(T const & x){
    switch(dtype_){
      case INT32_TYPE: value_.int32 = (int32_t)x; break;
      case FLOAT_TYPE: value_.float32 = (float)x; break;
      case DOUBLE_TYPE: value_.float64 = (double)x; break;
      default: throw;
    }
  }

public:
#define ISAAC_INSTANTIATE(TYPE) scalar(TYPE value, DType dtype = to_DType<TYPE>::value) : dtype_(dtype) { init(value); }
  ISAAC_INSTANTIATE(float)
  ISAAC_INSTANTIATE(double)
#undef ISAAC_INSTANTIATE

  void* data() const{
    switch(dtype_){
      case INT32_TYPE: return (void*)&value_.int32;
      case FLOAT_TYPE: return (void*)&value_.float32;
      case DOUBLE_TYPE: return (void*)&value_.float64;
      default: throw;
    }
  }

  DType dtype() const{
    return dtype_;
  }

private:
  DType dtype_;
  union{
    int32_t int32;
    float float32;
    double float64;
  }value_;
};

}

#endif
