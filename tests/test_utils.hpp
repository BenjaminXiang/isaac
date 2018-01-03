#include <cmath>
#include <vector>
#include <iostream>

double max_rounding_error(int32_t){ return 0; }
double max_rounding_error(double x){ return std::pow(2, int(std::log2(x)) - 52); }
double max_rounding_error(float x){ return std::pow(2, int(std::log2(x)) - 23); }
double max_rounding_error(half_float::half x){ return std::pow(2, int(std::log2(x)) - 10); }

template<class T>
bool is_correct(std::vector<T> const & iO, std::vector<T> const & rO, double eps){

  if(iO.size() != rO.size()){
    std::cout << "inputs don't have the same size" << std::endl;
    return false;
  }
  for(size_t i = 0 ; i < iO.size(); ++i){
    T io = iO[i], ro = rO[i];
    T denom = std::max(std::fabs(io), std::fabs(ro));
    denom = (denom==0)?1:denom;
    if(std::fabs(io - ro)/denom > eps || std::isnan(io)){
      std::cout << "idx " << i << ": " <<  io << " != " << ro << std::endl;
      return false;
    }
  }
  return true;
}
