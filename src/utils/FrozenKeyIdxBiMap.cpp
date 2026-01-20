/*! \field
 *  Implementation for FrozenKeyIdxBiMap
 */

#include "FrozenKeyIdxBiMap.h"

/*! Defines a bunch of extra details for implementing \ref FrozenKeyIdxBiMap */
namespace utils
{

namespace bimap_detail
{

/*! represents the result of an internal search for a key */
struct SearchRslt {
  std::optional<int> val;  ///< value found by the search
  int probe_count;         ///< number of probes before the search returned
  int rowidx;              ///< index of the "row" corresponding to the search result
};

void overwrite_row(Row& row, std::string key, uint16_t value)
{
  CHOLLA_ASSERT(row.keylen == 0, "Sanity check failed!");
  uint16_t keylen       = key.size();
  std::size_t total_len = keylen + 1;  // <- add 1 to account for '\0'
  char* key_ptr         = new char[total_len];
  std::memcpy(key_ptr, key.data(), total_len);
  row = Row{value, keylen, key_ptr};
}

SearchRslt search_helper_(const Row* rows, const char* key, int capacity, int max_probe,
                          const std::optional<HashRsltPack>& h)
{
  max_probe = (max_probe <= 0 || max_probe > capacity) ? capacity : max_probe;

  int i               = -1;  // <- set to a dummy value
  int launched_probes = 0;
  if (h.has_value() && h->keylen > 0 && max_probe > 0) {
    int keylen  = h->keylen;
    int guess_i = static_cast<int>(h->hash % capacity);  // <- initial guess

    do {  // circularly loop over rows to search for key (start at guess_i)
      i = (guess_i + launched_probes) % capacity;
      launched_probes++;  // <- about to perform a new probe
      const Row& r = rows[i];

      if (r.keylen == keylen && std::memcmp(r.key, key, keylen) == 0) {
        return SearchRslt{r.value, launched_probes, i};  // match found!
      }

      // check if rows[i] is empty or if we have hit the limit on searches
    } while (rows[i].keylen != 0 && launched_probes < max_probe);
  }

  return SearchRslt{std::nullopt, launched_probes, i};
}

/*! Search for the row matching key. The search ends when a match is found, an
 *  an empty row is found, or the function has probed `max_probe` entries
 *
 *  @param rows an array of rows to search to be compared
 *  @param key the key to be compared
 *  @param capacity the length of the rows array
 *  @param max_probe the maximum number of rows to check before giving up
 *
 *  @important
 *  The behavior is undefined if @p key is a `nullptr` or @p keylen is 0.
 */
SearchRslt search(const Row* rows, const char* key, int capacity, int max_probe)
{
  CHOLLA_ASSERT(key != nullptr, "Major programming oversight");
  std::optional<HashRsltPack> h = FNV1aHasher<>::calc(key);
  return search_helper_(rows, key, capacity, max_probe, h);
}

SearchRslt search(const Row* rows, std::string_view key, int capacity, int max_probe)
{
  std::optional<HashRsltPack> h = FNV1aHasher<>::calc(key);
  return search_helper_(rows, key.data(), capacity, max_probe, h);
}

}  // namespace bimap_detail

FrozenKeyIdxBiMap::FrozenKeyIdxBiMap(const std::vector<std::string>& keys) noexcept : FrozenKeyIdxBiMap()
{
  std::size_t n_keys = keys.size();
  if (n_keys == 0) {
    return;
  }
  CHOLLA_ASSERT(n_keys <= MAX_LEN, "too many keys were specified");
  // reminder: length = LOAD_FACTOR * capacity
  //           length = (LOAD_FACTOR_NUMERATOR / LOAD_FACTOR_DENOMINATOR) * capacity
  std::size_t capacity = (n_keys * LOAD_FACTOR_DENOMINATOR) / LOAD_FACTOR_NUMERATOR;
  CHOLLA_ASSERT(capacity > 0, "sanity check failed!")

  // let's validate the keys
  for (std::size_t i = 0; i < n_keys; i++) {
    CHOLLA_ASSERT(keys[i].size() > 0, "each key must hold at least 1 character");
    CHOLLA_ASSERT(keys[i].size() <= MAX_KEY_LEN, "no key may have more than %d characters", MAX_KEY_LEN);
    for (std::size_t j = 0; j < i; j++) {
      CHOLLA_ASSERT(keys[i] != keys[j], "\"%s\" key repeats", keys[i].c_str());
    }
  }

  // allocate hash table (each row is default constructed)
  bimap_detail::Row* rows = new bimap_detail::Row[capacity];
  // define a callback function to use when this is destroyed
  auto table_deleter = [capacity](bimap_detail::Row* rows) {
    for (std::size_t i = 0; i < capacity; i++) {
      if (rows[i].key != nullptr) {
        delete[] rows[i].key;
      }
    }
    delete[] rows;
  };
  this->table_rows_             = std::shared_ptr<bimap_detail::Row[]>(rows, table_deleter);
  uint16_t* ordered_row_indices = new uint16_t[n_keys];
  this->ordered_row_indices_    = std::shared_ptr<uint16_t[]>(ordered_row_indices, std::default_delete<uint16_t[]>());

  this->length_   = n_keys;
  this->capacity_ = capacity;

  // now it's time to fill in the array
  int max_probe_count = 1;
  for (std::size_t i = 0; i < n_keys; i++) {
    // search for the first empty row
    bimap_detail::SearchRslt search_rslt = bimap_detail::search(this->table_rows_.get(), keys[i], capacity, capacity);
    // this should be infallible (especially after we already did some checks)
    CHOLLA_ASSERT(search_rslt.probe_count != 0, "sanity check failed");

    // now we overwrite the row
    bimap_detail::overwrite_row(this->table_rows_.get()[search_rslt.rowidx], keys[i], i);
    this->ordered_row_indices_[i] = search_rslt.rowidx;

    max_probe_count = std::max(max_probe_count, search_rslt.probe_count);
  }
  this->max_probe_ = max_probe_count;
}

std::optional<int> FrozenKeyIdxBiMap::find(const char* key) const noexcept
{
  return bimap_detail::search(table_rows_.get(), key, capacity_, max_probe_).val;
}

std::optional<int> FrozenKeyIdxBiMap::find(std::string_view key) const noexcept
{
  return bimap_detail::search(table_rows_.get(), key, capacity_, max_probe_).val;
}

}  // namespace utils