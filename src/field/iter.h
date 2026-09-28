/*! \file
 *  Declares machinery associated with iterating over fields
 */
#pragma once

#include <cstdlib>   // std::size_t
#include <iterator>  // std::input_iterator_tag

#include "field_id.h"

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

namespace field
{

/*! \brief implements a C++ style InputIterator for iterating over \ref FieldIds
 *
 *  This exists so we can implement range-based for-loops iterate over subsets of
 *  FieldIds that share a common property
 *
 *  \note
 *  We could significantly simplify this implementation we tracked face-centered
 *  bfields and passively-advected scalars in separate field-packs from other conserved
 *  variables
 */
struct Iterator {
  // define type aliases that make up the common interface expected by C++ function
  // templates in the standard library for conveying properties about the iterator.
  using iterator_category = std::input_iterator_tag;
  using value_type        = FieldId;
  using difference_type   = std::ptrdiff_t;
  // (we skip defining a type alias called `pointer`)
  using reference = FieldId;  // <- type returned by deref & convertable to value_type

 private:
  uint8_t pack_id                      = 0;        ///< pack_id of each visited FieldId
  const field_detail::IdxSlc* slot_slc = nullptr;  ///< pointer to the slice objects
  uint8_t slot_offset                  = 0;        ///< current offset from slot_slc->start()

 public:
  Iterator(uint8_t pack_id, const field_detail::IdxSlc* slot_slc) : pack_id(pack_id), slot_slc(slot_slc), slot_offset(0)
  {
  }

  // overload `!=` and `==` operations
  bool operator!=(const Iterator& other) const { return !(*this == other); }
  bool operator==(const Iterator& o) const
  {
    return (pack_id == o.pack_id) && (slot_slc == o.slot_slc) && (slot_offset == o.slot_offset);
  }

  /*! \brief implements the dereference operation */
  reference operator*() const { return FieldId(pack_id, static_cast<uint8_t>(slot_slc->start() + slot_offset)); }

  /*! \brief implements the prefix increment operation
   *
   *  This effectively implements `++x`, which increments the value of `x`
   *  and returns the value of `x` from **after** after the increment */
  Iterator& operator++()
  {
    slot_offset++;
    if (slot_offset == slot_slc->size()) {
      slot_slc++;
      slot_offset = 0;
    }
    return *this;
  }

  /*! \brief implements the prefix increment operation
   *
   *  This effectively implements `x++`, which increments the value of `x`
   *  and returns a copy of `x` from **before** the increment */
  Iterator operator++(int)
  {
    Iterator ret = *this;
    ++(*this);
    return ret;
  }
};

/*! This is a "range" in the C++ 20 sense
 *
 *  See \ref FieldInfo::get_id_range for an example
 */
class IdRange
{
  uint8_t pack_id_ = 0;  // the common pack id of all fields in the "range"
  field_detail::IdxSlc slot_slc_seq_[2];
  int n_slc_ = 0;

 public:
  using iterator = Iterator;

  IdRange(uint8_t pack_id, field_detail::IdxSlc slc)
  {
    pack_id_         = pack_id;
    slot_slc_seq_[0] = slc;
    n_slc_           = (slot_slc_seq_[0].size() > 0) ? 1 : 0;
  }

  IdRange(uint8_t pack_id, field_detail::IdxSlc slc_a, field_detail::IdxSlc slc_b)
  {
    pack_id_ = pack_id;
    CHOLLA_ASSERT(slc_a.size() > 0, "when chaining to slices, the first slice should have at least 1 index");
    slot_slc_seq_[0] = slc_a;
    slot_slc_seq_[1] = slc_b;
    n_slc_           = (slot_slc_seq_[1].size() > 0) ? 2 : 1;
  }

  int n_items() const { return static_cast<int>(slot_slc_seq_[0].size() + slot_slc_seq_[1].size()); }

  Iterator begin() const { return Iterator(pack_id_, slot_slc_seq_); }
  Iterator end() const { return Iterator(pack_id_, slot_slc_seq_ + n_slc_); }
};

}  // namespace field