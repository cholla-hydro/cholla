/*! \file
 *  Holds tests for \ref TabulatedDynamicalDarkEnergyEoS
 */
#include <sstream>
#include <string>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

#include "tabulated_dynamicalDE_EoS.h"

// this is the path we use when we construct a TabulatedDynamicalDarkEnergyEoS directly
// from a string
const char* string_file_path_("dummy-file-path");

/*! A helper function that constructs a \ref TabulatedDynamicalDarkEnergyEoS instance
 *  by treating \p contents as the contents of a file
 */
TabulatedDynamicalDarkEnergyEoS construct_from_string_(const std::string& contents)
{
  std::istringstream file_contents(contents);
  // we pass silent=true to suppress informational messages summarizing properties of
  // the file when we read it
  TabulatedDynamicalDarkEnergyEoS dynamical_eos(string_file_path_, dynamic_cast<std::istream*>(&file_contents), true);
  return dynamical_eos;
}

/*! Encapsulates a sample input file that will be used to read in
 *  \ref TabulatedDynamicalDarkEnergyEOS
 */
struct FileVariation {
  std::string description;
  std::string content;
};

// teach GoogleTest how to print File Variation
void PrintTo(const FileVariation& fv, std::ostream* os) { *os << fv.description; }

// -------------------------------------------------------------------------------------

// we are going to run a simple test case where we try to parse variants of the
// following string
const std::string GOOD_CONTENTS_ = R"LITERAL(# z, w
0.000000000000000000e+00 -9.767616499273475972e-01
6.938631476027579126e-03 -9.769941793369097960e-01
# this is a random meaningless comment!
1.392540755881421788e-02 -9.772263520514015145e-01)LITERAL";

class tALLTabulatedDynamicalDarkEnergyEoS : public testing::TestWithParam<FileVariation>
{
};

TEST_P(tALLTabulatedDynamicalDarkEnergyEoS, SimpleOpen)
{
  TabulatedDynamicalDarkEnergyEoS dynamical_eos = construct_from_string_(GetParam().content);

  // the value at z=0 is always normalized to be 1.0
  EXPECT_EQ(dynamical_eos.Get_DynamicalDE_Density_from_a(1.0), 1.0);

  // check that the value before the earliest redshfift are all the same
  EXPECT_EQ(dynamical_eos.Get_DynamicalDE_Density_from_a(0.1), dynamical_eos.Get_DynamicalDE_Density_from_a(0.01));

  // it would be great to actually make some strong checks rather than just checking
  // the bounds (but that would require a more detailed understanding of the
  // calculation than I currently have)
}

const FileVariation valid_variants_[3] = {
    {"NoTerminalNewline", GOOD_CONTENTS_},
    {"WithTerminalNewline", GOOD_CONTENTS_ + '\n'},
    {"EmptyFinalLine", GOOD_CONTENTS_ + "\n\n"},
};

INSTANTIATE_TEST_SUITE_P(
    /* 1st arg intentionally empty */, tALLTabulatedDynamicalDarkEnergyEoS, testing::ValuesIn(valid_variants_),
    testing::PrintToStringParamName());

// -------------------------------------------------------------------------------------

// if we ever decide that DeathTests are too expensive, we can transition to using exceptions
// in the following tests

TEST(tALLTabulatedDynamicalDarkEnergyEoSDeathTest, InvalidPath)
{
  std::string path = "not/a/real/path.txt";
  ASSERT_DEATH({ TabulatedDynamicalDarkEnergyEoS dyanmical_eos(path); },
               "Unable to open DE equation of state file: " + path);
}

TEST(tALLTabulatedDynamicalDarkEnergyEoSDeathTest, MissingRedshift0)
{
  const std::string contents = R"LITERAL(# z, w
  6.938631476027579126e-03 -9.769941793369097960e-01
  1.392540755881421788e-02 -9.772263520514015145e-01)LITERAL";

  ASSERT_DEATH({ TabulatedDynamicalDarkEnergyEoS dyanmical_eos = construct_from_string_(contents); },
               "We require z_min = 0 so that w\\(z=0\\) is well defined");
}

TEST(tALLTabulatedDynamicalDarkEnergyEoSDeathTest, OutOfOrder)
{
  const std::string contents = R"LITERAL(# z, w
0.000000000000000000e+00 -9.767616499273475972e-01
1.392540755881421788e-02 -9.772263520514015145e-01
6.938631476027579126e-03 -9.769941793369097960e-01)LITERAL";

  ASSERT_DEATH(
      { TabulatedDynamicalDarkEnergyEoS dyanmical_eos = construct_from_string_(contents); },
      "ERROR: equation of state must be ordered such that redshift is increasing as the rows increase in the file");
}

TEST(tALLTabulatedDynamicalDarkEnergyEoSDeathTest, SingleElemOnLine2)
{
  const std::string contents = R"LITERAL(# z, w
0.000000000000000000e+00 
1.392540755881421788e-02 -9.772263520514015145e-01)LITERAL";

  ASSERT_DEATH({ TabulatedDynamicalDarkEnergyEoS dyanmical_eos = construct_from_string_(contents); },
               string_file_path_ + std::string(":2 doesn't specify 2 elements"));
}

TEST(tALLTabulatedDynamicalDarkEnergyEoSDeathTest, ThreeElemsOnLine2)
{
  const std::string contents = R"LITERAL(# z, w
0.0 -0.97 342
1.392540755881421788e-02 -9.772263520514015145e-01)LITERAL";

  ASSERT_DEATH({ TabulatedDynamicalDarkEnergyEoS dyanmical_eos = construct_from_string_(contents); },
               string_file_path_ + std::string(":2 doesn't specify 2 elements"));
}

// -------------------------------------------------------------------------------------

// define parametrized tests where we try to read files without real contents

class tALLTabulatedDynamicalDarkEnergyEoSNoContentsDeathTest : public testing::TestWithParam<FileVariation>
{
};

TEST_P(tALLTabulatedDynamicalDarkEnergyEoSNoContentsDeathTest, Simple)
{
  ASSERT_DEATH({ TabulatedDynamicalDarkEnergyEoS dynamical_eos = construct_from_string_(GetParam().content); },
               string_file_path_ + std::string(" doesn't contain any data"));
}

const FileVariation invalid_variants_[] = {
    {"Empty", ""},
    {"SingleBlankLine", "\n"},
    {"SingleComment", "# this is a comment!"},
};

INSTANTIATE_TEST_SUITE_P(
    /* 1st arg intentionally empty */, tALLTabulatedDynamicalDarkEnergyEoSNoContentsDeathTest,
    testing::ValuesIn(invalid_variants_), testing::PrintToStringParamName());