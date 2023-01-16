// STL Includes
#include <chrono>
#include <random>
#include <string>

// Local includes
#include "../global/global.h"

#pragma once

class ChollaPrngGenerator
{
 public:
  std::mt19937_64 inline static generator;

  ChollaPrngGenerator(struct parameters *P)
  {
    // If the seed isn't defined in the settings file or argv then generate
    // a random seed
    if (P->prng_seed == 0) {
      // Since std::random_device isn't guaranteed to be random or
      // different for each rank we're going to convert both the base seed
      // and MPI rank to strings, concatenated them, then hash the result.
      // This should give a fairly random seed even if std::random_device
      // isn't random
      std::string hashString =
          std::to_string(std::random_device{}())
#ifdef MPI_CHOLLA
          + std::to_string(static_cast<std::uint_fast64_t>(procID))
#endif
          + std::to_string(std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count());
      std::size_t hashedSeed = std::hash<std::string>{}(hashString);
      P->prng_seed           = static_cast<std::uint_fast64_t>(hashedSeed);
    }

    // Initialize the PRNG
    generator.seed(P->prng_seed);
  };
  ~ChollaPrngGenerator() = default;
};
