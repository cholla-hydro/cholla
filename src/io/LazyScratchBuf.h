/*! \file
 *  Defines the LazyScratchBuf type
 */

#include <type_traits>
#include <vector>

#include "../utils/DeviceVector.h"
#include "../utils/error_handling.h"

namespace io
{

namespace io_detail
{

template <class T>
struct TChecker {
  static_assert(!(std::is_pointer_v<T> || std::is_reference_v<T>), "T should not be a pointer or reference");
  static_assert(!(std::is_volatile_v<T> || std::is_const_v<T>), "T should not be volatile or const");
  typedef T type;
};

template <class T>
using CheckedT = typename TChecker<T>::type;

template <typename VecType>
typename VecType::value_type* resize_and_get_(VecType& buf, std::size_t buf_size)
{
  CHOLLA_ASSERT(buf_size > 0, "buf_size must be positive");
  if (buf.size() < buf_size) {
    buf.resize(buf_size);
  }
  return buf.data();
}

}  // namespace io_detail

/*! This is used for tracking lazily initialized scratch buffers (for io-purposes)
 *
 *  This is **NOT** threadsafe (but that's ok since many parts of Cholla aren't
 *  threadsafe)
 */
class LazyScratchBuf
{
  cuda_utilities::DeviceVector<float> d_f32_buf;
  cuda_utilities::DeviceVector<double> d_f64_buf;
  std::vector<float> h_f32_buf;
  std::vector<double> h_f64_buf;

 public:
  // prevent accidental deep copies (I can't imagine ever wanting this)

  LazyScratchBuf()                                 = default;
  LazyScratchBuf(const LazyScratchBuf&)            = delete;
  LazyScratchBuf& operator=(const LazyScratchBuf&) = delete;

  template <typename T>
  io_detail::CheckedT<T>* get_buf_dev(std::size_t buf_size)
  {
    if constexpr (std::is_same_v<T, float>) {
      return io_detail::resize_and_get_(this->d_f32_buf, buf_size);
    } else if constexpr (std::is_same_v<T, double>) {
      return io_detail::resize_and_get_(this->d_f64_buf, buf_size);
    } else {
      CHOLLA_ERROR("unrecognized type");
    }
  }

  template <typename T>
  io_detail::CheckedT<T>* get_buf_host(std::size_t buf_size)
  {
    if constexpr (std::is_same_v<T, float>) {
      return io_detail::resize_and_get_(this->h_f32_buf, buf_size);
    } else if constexpr (std::is_same_v<T, double>) {
      return io_detail::resize_and_get_(this->h_f64_buf, buf_size);
    } else {
      CHOLLA_ERROR("unrecognized type");
    }
  }
};

}  // namespace io