//
// Created by jiashuai on 18-1-16.
//

#ifndef THUNDERGBM_COMMON_H
#define THUNDERGBM_COMMON_H

#include "thundergbm/util/log.h"
#include "cuda_runtime_api.h"
#include "cstdlib"
#include "config.h"
#include "thrust/tuple.h"

#include "fasthb.cuh"
#include "fasthb/impl/build.cuh"

using std::vector;
using std::string;

//CUDA macro
#define USE_CUDA
#define NO_GPU \
LOG(FATAL)<<"Cannot use GPU when compiling without GPU"
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (false)

//https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

//data types
typedef double float_type;

#define HOST_DEVICE __host__ __device__

struct GHPair {
    float_type g;
    float_type h;

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g + rhs.g;
        res.h = this->h + rhs.h;
        return res;
    }

    HOST_DEVICE const GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g - rhs.g;
        res.h = this->h - rhs.h;
        return res;
    }

    HOST_DEVICE bool operator==(const GHPair &rhs) const {
        return this->g == rhs.g && this->h == rhs.h;
    }

    HOST_DEVICE bool operator!=(const GHPair &rhs) const {
        return !(*this == rhs);
    }

    HOST_DEVICE GHPair() : g(0), h(0) {};

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {};

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {};

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

template<>
class fasthb::build::Op<GHPair, GHPair>
{
public:
  using O = volatile GHPair;
  using I = GHPair;

  template<aggregate::Method method>
  __host__ __device__
  X_INL static void Aggregate(volatile O& dst, const I& src)
  {
#ifndef __CUDA_ARCH__
#if __cplusplus >= 202002L
    std::atomic<float_type> atomic_g{dst.g};
    std::atomic<float_type> atomic_h{dst.h};

    (void)atomic_g.fetch_add(
        static_cast<float_type>(src.g), std::memory_order_release);
    dst.g = atomic_g.load(std::memory_order_acquire);

    (void)atomic_h.fetch_add(
        static_cast<float_type>(src.h), std::memory_order_release);
    dst.h = atomic_h.load(std::memory_order_acquire);
#else
    printf("Atomic operations for floating numbers are not implemented before C++20.\n");
#endif  // __cplusplus
#else
    if (src.g != static_cast<float_type>(0)) {
      (void)atomicAdd(
          const_cast<float_type*>(&dst.g), static_cast<float_type>(src.g));
    }
    if (src.h != static_cast<float_type>(0)) {
      (void)atomicAdd(
          const_cast<float_type*>(&dst.h), static_cast<float_type>(src.h));
    }
#endif
  }

  template<binning::Method method>
  __host__ __device__
  X_INL static void Binning(index_t& index, const I& input, void* rest)
  {
    auto param{static_cast<binning::GivenIndexParam*>(rest)};
    binning::GivenIndex(index, param->iteration, param->indices, param->length);
  }

  __host__ __device__
  X_INL static void GetBinWidth(
      I& bin_width, const index_t& num_bin, const I& lower, const I& upper)
  {
  }

  template<typename O>
  __host__ __device__
  X_INL static void Reset(O& dst)
  {
    dst.g = static_cast<float_type>(0);
    dst.h = static_cast<float_type>(0);
  }

  template<bool atomic, typename O>
  __host__ __device__
  X_INL static void Reduce(volatile O& dst, const O& src)
  {
    if constexpr (atomic) {
#ifndef __CUDA_ARCH__
#if __cplusplus >= 202002L
      std::atomic<float_type> atomic_g{dst.g};
      std::atomic<float_type> atomic_h{dst.h};

      (void)atomic_g.fetch_add(
          static_cast<float_type>(src.g), std::memory_order_release);
      dst.g = atomic_g.load(std::memory_order_acquire);

      (void)atomic_h.fetch_add(
          static_cast<float_type>(src.h), std::memory_order_release);
      dst.h = atomic_h.load(std::memory_order_acquire);
#else
#endif  // __cplusplus
      printf("Atomic operations for floating numbers are not implemented before C++20.\n");
#else
      if (src.g != static_cast<float_type>(0)) {
        (void)atomicAdd(
            const_cast<float_type*>(&dst.g), static_cast<float_type>(src.g));
      }
      if (src.h != static_cast<float_type>(0)) {
        (void)atomicAdd(
            const_cast<float_type*>(&dst.h), static_cast<float_type>(src.h));
      }
#endif
    } else {
      dst = dst + src;
    }
  }
};  // namespace fasthb::build::Op<GHPair, GHPair>

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);

struct GBMParam {
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    float column_sampling_rate;
    std::string path;
    int verbose;
    bool profiling;
    bool bagging;
    int n_parallel_trees;
    float learning_rate;
    std::string objective;
    int num_class;
    int tree_per_rounds; // #tree of each round, depends on #class

    //for histogram
    int max_num_bin;
    float base_score=0;

    int n_device;

    std::string tree_method;
};
#endif //THUNDERGBM_COMMON_H
