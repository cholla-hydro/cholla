/*! \file
 *  Declares the FrozenKeyIdxBiMap type
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "error_handling.h"

namespace utils
{

namespace bimap_detail
{

/*! Holds the result of a call to fnv1a_hash */
struct HashRsltPack {
  std::uint16_t keylen;
  std::uint32_t hash;
};

/// implement equality operation (primarily for unit-testing)
inline bool operator==(const HashRsltPack& a, const HashRsltPack& b)
{
  return a.keylen == b.keylen && a.hash == b.hash;
}

/*! collects methods for computing a key's length and 32-bit FNV-1a hash.
 *
 *  \tparam MaxKeyLen the max number of characters in key (excluding '\0'). By default, it's the largest value
 *      \ref HashRsltPack::keylen holds. A smaller value can be specified as an optimization.
 *
 *  \note
 *  This hash function prioritizes convenience. We may want to evaluate whether alternatives (e.g. fxhash) are
 *  faster or have fewer collisions with our typical keys.
 */
template <int MaxKeyLen = std::numeric_limits<std::uint16_t>::max()>
struct FNV1aHasher {
  static_assert(0 <= MaxKeyLen && MaxKeyLen <= std::numeric_limits<std::uint16_t>::max(),
                "MaxKeyLen can't be encoded by HashRsltPack");

  inline static constexpr uint32_t FNV1A_PRIME  = 16777619;
  inline static constexpr uint32_t FNV1A_OFFSET = 2166136261;

  /*! Calculate the hash value
   *  \param key the null-terminated string. Behavior is deliberately undefined when passed a `nullptr`
   */
  static std::optional<HashRsltPack> calc(const char* key)
  {
    std::uint32_t hash = FNV1A_OFFSET;
    for (int i = 0; i <= MaxKeyLen; i++) {  // the `<=` is intentional
      if (key[i] == '\0') {
        return {HashRsltPack{static_cast<uint16_t>(i), hash}};
      }
      hash = (hash ^ key[i]) * FNV1A_PRIME;
    }
    return std::nullopt;
  }

  // this mostly exists as a convenience
  static std::optional<HashRsltPack> calc(std::string_view key)
  {
    int len = key.size();
    if (len > MaxKeyLen) {
      return std::nullopt;
    }
    std::uint32_t hash = FNV1A_OFFSET;
    for (int i = 0; i < len; i++) {
      hash = (hash ^ key[i]) * FNV1A_PRIME;
    }
    return {HashRsltPack{static_cast<uint16_t>(len), hash}};
  }
};

/*! A hash table is just an array of `Row`s, which are (key,value) pair with a little extra metadata.
 */
struct Row {
  // smallest structs members are listed first to minimize struct size
  uint16_t value  = 0;        ///< value associated with the current key
  uint16_t keylen = 0;        ///< length of the key (not including the '\0')
  const char* key = nullptr;  ///< identifies the address of this entry's key
};

}  // namespace bimap_detail

/*! @brief A bidirectional map (bimap), specialized to map `n` unique string keys to unique indexes with values
 *  of `0` through `(n-1)` and vice versa. The ordering & values of keys are set at creation and frozen.
 *
 *  @par Why Frozen?
 *  The contents are "frozen" for 3 primary reasons:
 *  1. It drastically simplifies the implementation (we don't have to worry about deletion -- which can be quite messy)
 *  2. Linear-probing generally provides better data locality than other hash collision resolution techniques, but
 *     generally has other drawbacks. Freezing the contents lets us mitigate many drawbacks (mostly related to
 *     the deletion operation)
 *  3. It let's us make copy operations cheaper. Since we know the map won't change, we
 *     can just use reference counting.
 *
 * @par
 * I would be stunned if `std::map<std::string, uint16_t>` or `std::map<const char*, uint16_t>` is faster than the
 * internal hash table since `std::map` is usually implemented as a tree.
 */
class FrozenKeyIdxBiMap
{
  // define attributes:

  /*! actual hash table data */
  std::shared_ptr<bimap_detail::Row[]> table_rows_;
  /*! tracks the row indices to make reverse lookups fast */
  std::shared_ptr<uint16_t[]> ordered_row_indices_;
  /*! number of table rows */
  uint16_t capacity_;
  /*! number of contained strings */
  uint16_t length_;
  /*! max number of rows that must be probed to determine if a key is contained */
  uint16_t max_probe_;

  // define a few constants

  /*! the load factor specifies the fraction of the capacity of the Hash table that is filled. */
  inline static constexpr int LOAD_FACTOR_NUMERATOR   = 2;
  inline static constexpr int LOAD_FACTOR_DENOMINATOR = 3;
  static_assert(LOAD_FACTOR_NUMERATOR <= LOAD_FACTOR_DENOMINATOR);

  inline static constexpr int64_t MAX_CAPACITY = static_cast<int64_t>(std::numeric_limits<uint16_t>::max());
  inline static constexpr int64_t MAX_LEN      = MAX_CAPACITY * LOAD_FACTOR_NUMERATOR / LOAD_FACTOR_DENOMINATOR;

  /*! specifies maximum allowed length of a key (excluding the null terminator).
   *
   *  \note
   *  I don't think we really need a key length that is much longer than this, and it would be useful to start enforcing
   * it so we can preserve an opportunity for a particular opimization:
   *  - essentially, if this is small enough, we could directly embed the characters of each key directly within each
   * `Row`. In practice, empty `Row`s will waste space (so we are motivated to keep this small)
   *  - this lets us reduce the number of pointer indirections while probing, and paves the way for fixed-sized
   *    memcmp evaluations.
   *  - Ideally, we want this to be 21 or 29 so that the total size of `Row` is a factor of 64. With a little extra work
   *    to align `table_rows_` to 64 bytes. That would help with cache locality (after all a cache line is 64 bytes),
   * and with a smidge more work, we could convince the compiler to autovectorize key-comparisons.
   */
  inline static constexpr uint16_t MAX_KEY_LEN = 21;

 public:  // Interface:
  /*! Default Constructor */
  FrozenKeyIdxBiMap() : table_rows_(nullptr), ordered_row_indices_(nullptr), capacity_(0), length_(0), max_probe_(0) {}

  /*! Main Constructor
   *
   *  \param keys Sequence of 1 or more unique strings.
   */
  explicit FrozenKeyIdxBiMap(const std::vector<std::string>& keys) noexcept;

  /*! Lookup the value associated with the specified key
   *
   *  This returns an empty optional if the key isn't known.
   */
  std::optional<int> find(const char* key) const noexcept;
  std::optional<int> find(std::string_view key) const noexcept;
  /*
  std::optional<int> find(const std::string& key) const noexcept {
    return this->find(std::string_view(key));
  }*/

  /*! Return the key associated with the specified value
   *
   *  For some context, if this function returns a string `s` for some index `i`, then a call to
   *  \ref FrozenKeyIdxBiMap::find that passes `s` will return `i`
   *
   * \warning
   * Invalid indices (i.e. `index < 0` OR `index >= length`) produce undefined behavior
   */
  std::string inverse_find(int index) const
  {
    uint16_t row_index = ordered_row_indices_.get()[index];
    return std::string(table_rows_.get()[row_index].key);
  }

  /*! return the number of keys in the map */
  std::size_t size() const noexcept { return length_; }
};

}  // namespace utils