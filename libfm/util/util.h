//
// Created by yanyu on 2017/7/4.
// 共用函数
//

#ifndef PYLIBFM_UTIL_H
#define PYLIBFM_UTIL_H

#include <vector>

#ifdef _WIN32
#include <float.h>
#else
#include <sys/resource.h>
#endif

#include <iostream>
#include <fstream>

typedef unsigned int uint;
typedef unsigned long long uint64;

#ifdef _WIN32
namespace std {
	bool isnan(double d) { return _isnan(d); }
	bool isnan(float f) { return  _isnan(f); }
	bool isinf(double d) { return (! _finite(d)) && (! isnan(d)); }
	bool isinf(float f) { return (! _finite(f)) && (! isnan(f)); }
}
#endif

#include <math.h>


template<typename T> inline T sqr(const T& d) { return d*d; }

template<typename T> inline T sigmoid(const T& d) { return (double)1.0/(1.0+exp(-d)); }

#endif //PYLIBFM_UTIL_H
