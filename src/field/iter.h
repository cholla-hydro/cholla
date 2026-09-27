/*! \file
 *  Declares machinery associating with iterating over field indices.
 */
#pragma once

#include <cstdlib>  // std::size_t

// define constructs that code outside of the field submodule never directly encounters
namespace field_detail
{

/*! \brief Encodes a contiguous slice of indices */
class IdxSlc
{
  std::size_t start_;
  std::size_t stop_;

 public:
  IdxSlc() : start_(0), stop_(0) {}

  IdxSlc(std::size_t start, std::size_t stop) : start_(start), stop_(stop)
  {
    CHOLLA_ASSERT(stop > start, "primary constructor requires at least 1 elem");
  }

  std::size_t start() const { return start_; }
  std::size_t stop() const { return stop_; }
  std::size_t size() const { return stop_ - start_; }
  bool contains(std::size_t idx) const { return (start_ <= idx) && (idx < stop_); }
};

}  // namespace field_detail