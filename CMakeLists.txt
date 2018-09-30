cmake_minimum_required(VERSION 3.3)
project(libfm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(SOURCE_FILES libfm/libfm_api.cpp libfm/util/util.h libfm/util/random.h libfm/util/matrix.h libfm/util/fmatrix.h libfm/core/sgd.h libfm/core/model.h libfm/libfm/learn.h libfm/libfm/learn_sgd.h libfm/libfm/learn_sgd_element.h libfm/libfm_api.cpp libfm/libfm/learn_sgd_element_adapt_reg.h libfm/libfm/learn_mcmc.h libfm/libfm/learn_mcmc_simultaneous.h)
add_library(libfm STATIC ${SOURCE_FILES})