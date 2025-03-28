#include <cstdio>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

#include "../io/ParameterMap.h"

namespace  // Anonymous namespace
{

struct DummyFile {
  /* This encapsulates a dummy parameter file. The function used to create the FILE object
   * ensures that the file is automatically deleted when it is closed, if the program exits
   * through std::exit, or by returning from main
   */
  std::FILE* fp;

  // be sure that the content isn't an empty string
  DummyFile(const char* content)
  {
    this->fp = std::tmpfile();
    // write the contents to the file
    if (std::fprintf(this->fp, "%s", content) < 0) {
      std::fprintf(stderr, "problem during fprintf\n");
      std::fclose(this->fp);
      std::exit(1);
    }
    if (std::fflush(this->fp) != 0) {
      std::fprintf(stderr, "problem during fflush\n");
      std::fclose(this->fp);
      std::exit(1);
    }
    // reset the position of fp back to start
    if (std::fseek(this->fp, 0, SEEK_SET) != 0) {
      std::fprintf(stderr, "problem during fprintf\n");
      std::fclose(this->fp);
      std::exit(1);
    };
  }

  ~DummyFile() { std::fclose(fp); }
};

}  // namespace

TEST(tALLParameterMap, Methodsize)
{
  const char* CONTENTS = R"LITERAL(
# My sample parameters
tout=50000
)LITERAL";

  DummyFile dummy   = DummyFile(CONTENTS);
  ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr);
  EXPECT_EQ(pmap.size(), 1);
}

const char* EXAMPLE_FILE_CONTENT_ = R"LITERAL(
# My sample parameter file
# ------------------------
tout=50000
outstep=100
gamma=1.4
# name of initial conditions
init=Disk_3D
n_hydro=10
xmin=-5
mypar=true
mypar2=false
)LITERAL";

TEST(tALLParameterMap, Methodhasparam)
{
  DummyFile dummy                            = DummyFile(EXAMPLE_FILE_CONTENT_);
  ParameterMap pmap                          = ParameterMap(dummy.fp, 0, nullptr);
  const std::vector<std::string> param_names = {"tout",    "outstep", "gamma", "init",
                                                "n_hydro", "xmin",    "mypar", "mypar2"};
  for (const std::string& param : param_names) {
    EXPECT_TRUE(pmap.has_param(param)) << "The has_param method should return True";
  }

  EXPECT_FALSE(pmap.has_param("notAparameter")) << "The has_param method should return False";
}

TEST(tALLParameterMap, Methodparamhastype)
{
  const char* CONTENTS = R"LITERAL(
# My sample parameters
# tout is large enough that it can't be represented by a 32-bit integer
tout=3000000000
gamma=1.4
init=Disk_3D
xmin=-5
mypar=true
)LITERAL";

  DummyFile dummy   = DummyFile(CONTENTS);
  ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr);
  std::string temp  = "tout";

  EXPECT_TRUE(pmap.param_has_type<std::int64_t>("tout"));
  if (sizeof(int) <= 4) {
    EXPECT_FALSE(pmap.param_has_type<int>("tout"));
  } else {
    EXPECT_TRUE(pmap.param_has_type<int>("tout"));
  }
  EXPECT_FALSE(pmap.param_has_type<bool>("tout"));
  EXPECT_TRUE(pmap.param_has_type<double>("tout"));
  EXPECT_TRUE(pmap.param_has_type<std::string>("tout"));

  EXPECT_FALSE(pmap.param_has_type<std::int64_t>("gamma"));
  EXPECT_FALSE(pmap.param_has_type<int>("gamma"));
  EXPECT_FALSE(pmap.param_has_type<bool>("gamma"));
  EXPECT_TRUE(pmap.param_has_type<double>("gamma"));
  EXPECT_TRUE(pmap.param_has_type<std::string>("gamma"));

  EXPECT_FALSE(pmap.param_has_type<std::int64_t>("init"));
  EXPECT_FALSE(pmap.param_has_type<int>("init"));
  EXPECT_FALSE(pmap.param_has_type<bool>("init"));
  EXPECT_FALSE(pmap.param_has_type<double>("init"));
  EXPECT_TRUE(pmap.param_has_type<std::string>("init"));

  EXPECT_TRUE(pmap.param_has_type<std::int64_t>("xmin"));
  EXPECT_TRUE(pmap.param_has_type<int>("xmin"));
  EXPECT_FALSE(pmap.param_has_type<bool>("xmin"));
  EXPECT_TRUE(pmap.param_has_type<double>("xmin"));
  EXPECT_TRUE(pmap.param_has_type<std::string>("xmin"));

  EXPECT_FALSE(pmap.param_has_type<std::int64_t>("mypar"));
  EXPECT_FALSE(pmap.param_has_type<int>("mypar"));
  EXPECT_TRUE(pmap.param_has_type<bool>("mypar"));
  EXPECT_FALSE(pmap.param_has_type<double>("mypar"));
  EXPECT_TRUE(pmap.param_has_type<std::string>("mypar"));
}

TEST(tALLParameterMap, Methodvalue)
{
  DummyFile dummy   = DummyFile(EXAMPLE_FILE_CONTENT_);
  ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr);
  EXPECT_EQ(pmap.value<std::int64_t>("tout"), 50000);
  EXPECT_EQ(pmap.value<int>("outstep"), 100);
  EXPECT_TRUE(pmap.value<double>("gamma") == std::stod("1.4"));
  EXPECT_TRUE(pmap.value<std::string>("init") == "Disk_3D");
  EXPECT_EQ(pmap.value<std::int64_t>("n_hydro"), 10);
  EXPECT_EQ(pmap.value<std::int64_t>("xmin"), -5);
  EXPECT_TRUE(pmap.value<bool>("mypar"));
  EXPECT_FALSE(pmap.value<bool>("mypar2"));

  // confirm that we accessed all parameters
  EXPECT_EQ(pmap.warn_unused_parameters({}, false, true), 0);
}

TEST(tALLParameterMap, Methodvalueor)
{
  DummyFile dummy   = DummyFile(EXAMPLE_FILE_CONTENT_);
  ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr);

  const std::size_t original_size = pmap.size();

  EXPECT_EQ(pmap.value_or("tout", 7), 50000);
  EXPECT_EQ(pmap.value_or("toutNOTREAL", 7), 7);

  std::int64_t dflt = -3424;
  EXPECT_EQ(pmap.value_or("tout", dflt), 50000);
  EXPECT_EQ(pmap.value_or("toutNOTREAL", dflt), dflt);

  EXPECT_TRUE(pmap.value_or("init", "not-real!") == "Disk_3D");
  EXPECT_TRUE(pmap.value_or("initNOTREAL", "not-real!") == "not-real!");

  EXPECT_TRUE(pmap.value_or("gamma", 1.0) == std::stod("1.4"));
  EXPECT_TRUE(pmap.value_or("gammaNOTREAL", 1.0) == 1.0);

  EXPECT_TRUE(pmap.value_or("mypar", false));
  EXPECT_FALSE(pmap.value_or("myparNOTREAL", false));

  EXPECT_FALSE(pmap.value_or("mypar2", true));
  EXPECT_TRUE(pmap.value_or("mypar2NOTREAL", true));

  // using value_or never mutates the contents.
  EXPECT_FALSE(pmap.has_param("mypar2NOTREAL"));
  EXPECT_EQ(original_size, pmap.size());
}

TEST(tALLParameterMap, Methodwarnunusedparameters)
{
  const char* CONTENTS = R"LITERAL(
# My sample parameters
tout=50000
gamma=1.4
mypar=true
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ParameterMap pmap    = ParameterMap(dummy.fp, 0, nullptr);

  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3) << "baseline case doesn't work";
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0)
      << "ignore_params argument doesn't work right";

  // using has_param - confirm this has no effect (whether it exists or not)
  EXPECT_FALSE(pmap.has_param("NOTREAL"));
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3);
  EXPECT_TRUE(pmap.has_param("tout"));
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0);

  // using param_has_type - confirm this has no effect (whether it exists or not)
  EXPECT_FALSE(pmap.param_has_type<std::string>("NOTREAL"));
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3);
  EXPECT_TRUE(pmap.param_has_type<std::string>("tout"));
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0);

  // using value_or (with missing value) - confirm this has no effect
  EXPECT_EQ(pmap.value_or("NOTREAL", 1), 1);
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 3);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0);

  // using value_or (on present value)
  EXPECT_TRUE(pmap.value_or("gamma", 1.0) == std::stod("1.4"));
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 2);
  ASSERT_EQ(pmap.warn_unused_parameters({"gamma"}, false, true), 2);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "mypar"}, false, true), 0);

  // using value (on present value)
  EXPECT_EQ(pmap.value<std::int64_t>("tout"), 50000);
  ASSERT_EQ(pmap.warn_unused_parameters({}, false, true), 1);
  ASSERT_EQ(pmap.warn_unused_parameters({"gamma", "tout"}, false, true), 1);
  ASSERT_EQ(pmap.warn_unused_parameters({"tout", "gamma", "mypar"}, false, true), 0);
  ASSERT_EQ(pmap.warn_unused_parameters({"mypar"}, false, true), 0);
}

// the test code for all of the examples is based on the TOML 1.0.0 specification
// https://toml.io/en/v1.0.0#table

TEST(tALLParameterMapTables, NoKeys)
{
  const char* CONTENTS = R"LITERAL(
[table]
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ParameterMap pmap    = ParameterMap(dummy.fp, 0, nullptr);
  ASSERT_EQ(pmap.size(), 0);
}

TEST(tALLParameterMapTables, simple)
{
  const char* CONTENTS = R"LITERAL(
[table-1]
key1=-1.0
key2=123

[table-2]
key1=-2.0
key2=456
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ParameterMap pmap    = ParameterMap(dummy.fp, 0, nullptr);
  ASSERT_EQ(pmap.size(), 4);
  pmap.Enforce_Table_Content_Uniform_Access_Status("table-1", true);
  ASSERT_EQ(pmap.value<double>("table-1.key1"), -1.0);
  ASSERT_EQ(pmap.value<int>("table-1.key2"), 123);
  pmap.Enforce_Table_Content_Uniform_Access_Status("table-1", false);

  pmap.Enforce_Table_Content_Uniform_Access_Status("table-2", true);
  ASSERT_EQ(pmap.value<double>("table-2.key1"), -2.0);
  ASSERT_EQ(pmap.value<int>("table-2.key2"), 456);
  pmap.Enforce_Table_Content_Uniform_Access_Status("table-2", false);
}

TEST(tALLParameterMapTables, EmptyName)
{
  const char* CONTENTS = R"LITERAL(
[]
key1=-1.0
key2=123
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr););
}

TEST(tALLParameterMapTables, BadNames)
{
  std::vector<std::string> bad_names = {"[.]",    "[ ]",        "[ fdssf]",   "[sadfasd ]", "[sda.]",
                                        "[.asd]", "[ssad.sd ]", "[ sdas.sd]", "[sds sdsd]"};
  for (const std::string& table_name : bad_names) {
    std::string contents = table_name + "\nkey1=-1.0\n";
    DummyFile dummy      = DummyFile(contents.c_str());
    ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr););
  }
}

TEST(tALLParameterMapTables, ForbidTableRedefine)
{
  const char* CONTENTS = R"LITERAL(
[fruit]
apple=1

[fruit]
orange=2
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr););
}

TEST(tALLParameterMapTables, DontRedefineTable)
{
  const char* CONTENTS = R"LITERAL(
[fruit]
apple=1

[fruit]
orange=2
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr););
}

TEST(tALLParameterMapTables, ForbidTableKeyNameCollision)
{
  const char* CONTENTS = R"LITERAL(
[fruit]
apple=1

[fruit.apple]
texture=2
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy.fp, 0, nullptr););
}

TEST(tALLParameterMapTables, TopLevelTableAndTable)
{
  const char* CONTENTS = R"LITERAL(
# these parameters are in the top-level table
key1=-1.0
key2=100.0

[table]
key1=-2.0
key2=456
)LITERAL";
  DummyFile dummy      = DummyFile(CONTENTS);
  ParameterMap pmap    = ParameterMap(dummy.fp, 0, nullptr);
  ASSERT_EQ(pmap.size(), 4);
  ASSERT_EQ(pmap.value<double>("key1"), -1.0);
  ASSERT_EQ(pmap.value<double>("key2"), 100.0);
  ASSERT_EQ(pmap.value<double>("table.key1"), -2.0);
  ASSERT_EQ(pmap.value<int>("table.key2"), 456);
}

// for now we forbid dotted keys from being specified inside a parameter file
// (the following cases are technically supported by TOML)
TEST(tALLParameterMapTables, TemporaryForbiddenDottedKeys)
{
  const char* CONTENTS1 = R"LITERAL(
fruit.apple.color=3
# Defines a table named fruit
# Defines a table named fruit.apple

fruit.apple.taste.sweet=true
# Defines a table named fruit.apple.taste
# fruit and fruit.apple were already created
)LITERAL";
  DummyFile dummy1      = DummyFile(CONTENTS1);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy1.fp, 0, nullptr););

  const char* CONTENTS2 = R"LITERAL(
[fruit]
apple.color=1
apple.taste.sweet=true

# you should be able to add sub-tables
[fruit.apple.texture]
smooth=true
)LITERAL";
  DummyFile dummy2      = DummyFile(CONTENTS2);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy2.fp, 0, nullptr););
}

// the following tests check cases that toml forbids related to dotted-keys
// (because these scenarios redefine tables)
TEST(tALLParameterMapTables, ForbiddenRedefineTableDottedKeys)
{
  const char* CONTENTS1 = R"LITERAL(
[fruit]
apple.color=1
apple.taste.sweet=true

[fruit.apple]
)LITERAL";
  DummyFile dummy1      = DummyFile(CONTENTS1);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy1.fp, 0, nullptr););

  const char* CONTENTS2 = R"LITERAL(
[fruit]
apple.color=1
apple.taste.sweet=true

[fruit.apple.taste]
)LITERAL";
  DummyFile dummy2      = DummyFile(CONTENTS2);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy2.fp, 0, nullptr););

  // it's a little less clear that this last case should correspond to a failure
  // based on the wording on the toml site, it sounds like it should be disallowed
  const char* CONTENTS3 = R"LITERAL(
[fruit.apple.taste]
sweet=true

[fruit]
apple.taste.soure=false
)LITERAL";
  DummyFile dummy3      = DummyFile(CONTENTS3);
  ASSERT_ANY_THROW(ParameterMap pmap = ParameterMap(dummy3.fp, 0, nullptr););
}