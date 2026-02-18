/*! \file DeviceVector_tests.cu
 *  \brief Tests for the FrozenKeyIdxBiMap class
 */

#include <gtest/gtest.h>

#include <iomanip>  // std::setfill, std::setw, std::hex
#include <ostream>  // std::ostream

#include "FrozenKeyIdxBiMap.h"

// =============================================================================
// Tests for the hash function
// =============================================================================

namespace utils::bimap_detail
{
/*! Teach GTest how to print HashRsltPack
 *  \note it's important this is in the same namespace as HashRsltPack */
void PrintTo(const HashRsltPack& pack, std::ostream* os)
{
  *os << "{keylen: " << pack.keylen << ", hash: 0x" << std::setfill('0') << std::setw(8)  // u32 has 8 hex digits
      << std::hex << pack.hash << "}";
}

}  // namespace utils::bimap_detail

// the test answers primarily came from Appendix C of
// https://datatracker.ietf.org/doc/html/draft-eastlake-fnv-17

using utils::bimap_detail::FNV1aHasher;
using utils::bimap_detail::HashRsltPack;

TEST(tALLBiMapFNV1a, EmptyString)
{
  std::optional<HashRsltPack> expected{{0, 0x811c9dc5ULL}};
  EXPECT_EQ(FNV1aHasher<>::calc(""), expected);
  EXPECT_EQ(FNV1aHasher<>::calc(std::string_view("")), expected);
}

TEST(tALLBiMapFNV1a, aString)
{
  std::optional<HashRsltPack> expected{{1, 0xe40c292cULL}};
  EXPECT_EQ(FNV1aHasher<>::calc("a"), expected);
  EXPECT_EQ(FNV1aHasher<>::calc(std::string_view("a")), expected);
}

TEST(tALLBiMapFNV1a, foobarString)
{
  std::optional<HashRsltPack> expected{{6, 0xbf9cf968ULL}};
  EXPECT_EQ(FNV1aHasher<>::calc("foobar"), expected);
  EXPECT_EQ(FNV1aHasher<>::calc(std::string_view("foobar")), expected);
}

TEST(tALLBiMapFNV1a, MaxSizeString)
{
  constexpr int MaxKeyLen = 6;  // <- exactly matches the key's length
  std::optional<HashRsltPack> expected{{6, 0xbf9cf968ULL}};
  EXPECT_EQ(FNV1aHasher<MaxKeyLen>::calc("foobar"), expected);
  EXPECT_EQ(FNV1aHasher<MaxKeyLen>::calc(std::string_view("foobar")), expected);
}

TEST(tALLBiMapFNV1a, TooLongString)
{
  constexpr int MaxKeyLen = 5;  // <- shorter than the queried key
  EXPECT_EQ(FNV1aHasher<MaxKeyLen>::calc("foobar"), std::nullopt);
  EXPECT_EQ(FNV1aHasher<MaxKeyLen>::calc(std::string_view("foobar")), std::nullopt);
}

// =============================================================================
// Miscellaneous Examples
// =============================================================================

TEST(tALLBiMapGeneral, FullExample)
{
  // THE SCENARIO: we have a list of unique ordered strings
  //
  // We are going build a FrozenKeyIdxBiMap instance from the following list.
  // The resulting object is a bidirectional map that can both:
  // 1. map a string to its index (at the time of construction) in the list.
  //    - example: "momentum_x" is mapped to 1
  //    - example: "GasEnergy" is mapped to 6
  // 2. perform the reverse mapping (i.e. index -> string)
  //    - example: 1 is mapped to "momentum_x"
  //    - example: 6 is mapped to "GasEnergy"
  //
  // It's worth emphasizing that the mapping is frozen when its constructed &
  // contents can't be changed (even if you reorder the original)
  std::vector<std::string> keys = {"density", "momentum_x",   "momentum_y", "momentum_z",
                                   "Energy",  "dust_density", "GasEnergy"};

  // PART 1: build a FrozenKeyIdxBiMap from this list
  utils::FrozenKeyIdxBiMap m(keys);

  // PART 2: let's show some examples of lookups from names

  // Equivalent Python:  `1 == m["momentum_x"]`
  EXPECT_EQ(m.find("momentum_x"), std::optional<int>{1});
  EXPECT_EQ(m.find(std::string("momentum_x")), std::optional<int>{1});
  EXPECT_EQ(m.find(std::string_view("momentum_x")), std::optional<int>{1});

  // Equivalent Python/idiomatic C++:  `6 == m["GasEnergy"]`
  EXPECT_EQ(m.find("GasEnergy"), std::optional<int>{6});
  EXPECT_EQ(m.find(std::string("GasEnergy")), std::optional<int>{6});
  EXPECT_EQ(m.find(std::string_view("GasEnergy")), std::optional<int>{6});

  // for unknown key, returns an empty optional
  EXPECT_EQ(m.find("Dummy"), std::nullopt);

  // PART 3: let's show the reverse of the previous lookups
  EXPECT_EQ(m.inverse_find(1), "momentum_x");
  EXPECT_EQ(m.inverse_find(6), "GasEnergy");

  // PART 4: We can also query the length
  EXPECT_EQ(m.size(), 7);
}

// validate basic operations for an empty bimap
TEST(tALLBiMapGeneral, EmptyBasicOps)
{
  utils::FrozenKeyIdxBiMap m;

  EXPECT_EQ(m.size(), 0) << "an empty mapping should have a size of 0";

  EXPECT_EQ(m.find("key"), std::nullopt) << "key lookup should always fail for an empty mapping";
}

TEST(tALLBiMapDeathTest, TooLongKey)
{
  std::string long_key(100, 'A');
  std::vector<std::string> keys = {"density", long_key};

  ASSERT_DEATH({ utils::FrozenKeyIdxBiMap m(keys); }, "no key may have more than .* characters");
}

TEST(tALLBiMapDeathTest, EmptyKey)
{
  std::vector<std::string> keys = {"density", ""};

  ASSERT_DEATH({ utils::FrozenKeyIdxBiMap m(keys); }, "each key must hold at least 1 character");
}

TEST(tALLBiMapDeathTest, DuplicateKey)
{
  std::vector<std::string> keys = {"density", "momentum_x", "density"};

  ASSERT_DEATH({ utils::FrozenKeyIdxBiMap m(keys); }, "\"density\" key repeats");
}