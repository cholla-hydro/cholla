# Testing

> *Note: Testing is still in beta and is missing some features, namely testing
> automation. If you spot any errors or bugs please create an
> [issue](https://github.com/cholla-hydro/cholla/issues) and we'll work on
> resolving it.*

If you are new to the testing world, check the bottom of this page for a glossary and links to various sites with more information.

## Building & Running Existing Tests

### Setup

The tests have three dependencies:

- GoogleTest, which can be can be installed as part of the testing scripts or
  loaded as a module if your system has it
- The [Cholla-tests-data](https://github.com/cholla-hydro/cholla-tests-data)
  submodule which in turn requires [Git LFS](https://git-lfs.github.com).
  - To download the cholla-tests-data submodule simply clone cholla then run
    `git submodule update --init` in the cholla repository. See the submodule
    [README](https://github.com/cholla-hydro/cholla-tests-data/blob/main/README.md)
    for more details.
  - If you don't have Git LFS available on your cluster then you can clone the
    repo on a computer with Git LFS, it's easy to install on personal computers,
    and copy the contents to the `cholla-tests-data` directory inside the Cholla
    repository

### Building & Launching with the Included Script

If you are on a supported machine, the simplest way to build and run tests is with the script {repository-file}`builds/scripts/run_tests.sh`.
That script contains functions to clean, source setup scripts, optionally
build GoogleTest, build Cholla, build the tests, and launch the tests. For example, using the script to build with GCC and run all hydro tests:

```bash
> cd cholla
> source builds/run_tests.sh
> buildAndRunTests -c gcc -t hydro
```

The machine name and launch command are automatically set based on the hostname. The
make type defaults to `hydro` but can be set to another type with the `-t` flag. The compiler is set with the `-c` flag; to use a different compiler replace `gcc` with the name of that compiler. Note that it must exactly match the name used in the setup file, i.e. on Summit you can use `xl`. The function `buildAndRunTests` runs all the other functions in the proper
order. Details can be found in the block comments above each function in the shell script.

To use `buildAndRunTests` you need to have GoogleTest available. You can either
load a GoogleTest module or install and compile it from source yourself. If
you load it with a module make sure the `GOOGLETEST_ROOT` environment variable
is pointing at the install path. If you compile it yourself then you can use the
`buildGoogleTest` function in `run_tests.sh` or pass `buildAndRunTests` the `-g`
flag.

### Building & Launching Manually

1. Check that GoogleTest is installed in your machine and load the module if
   applicable.
2. Check that the `GOOGLETEST_ROOT` variable is set in the Makefile for you
   machine. E.g. go to `cholla/builds/make.host.MACHINE_NAME` and assign
   `GOOGLETEST_ROOT` to be the path to your installation of GoogleTest.
   - An example can be found in {repository-file}`builds/make.host.summit`
3. Build Cholla as normal using whichever make type you want.
   This executable will be used to run system tests
   - Example: `make -j TYPE=hydro` builds the hydro make type.
4. Build the tests executable by running `make` again but now with the `TEST`
   make variable set to `true`. You *must* use the same make type as was used
   for building Cholla in the previous step.
   - Example: `make -j TYPE=hydro TEST=true`
5. Launch the tests: Tests should be launched within an interactive or batch job
   if you're running on a cluster as they can be very computationally intensive.
   Be sure to use the appropriate launch command for your system. Note that the tests
   themselves do not use MPI and the appropriate launch command for launching
   the system tests is given with the `--mpi-launcher` flag. Tests also require
   some command line arguments to run, these are listed here:
   - `--cholla-root path/to/cholla/root` - The full path to the Cholla
     directory. A relative path *might* work but that is untested and
     unverified.
   - `--build-type` - The make type used, e.g. `hydro`, `gravity`, etc.
   - `--machine` - The machine used. Must match the name returned by {repository-file}`builds/machine.sh`
   - A full launch command that runs all the hydro tests looks like:
     - `{chollaRoot}/bin/cholla.type.machine.tests --cholla-root {chollaRoot} --build-type hydro --machine spock --gtest_filter=*tHYDRO*`
   - A full launch command that runs only hydro system tests looks like:
     - `{chollaRoot}/bin/cholla.type.machine.tests --cholla-root {chollaRoot} --build-type hydro --machine spock --gtest_filter=*tHYDROSYSTEM*`
   - **Optional Flags**
     - `--mpi-launcher` - The mpi launcher to use. It *must* end in the
       parameter that indicates the number of ranks to launch and that parameter
       must be left blank, i.e.
       `jsrun --smpiargs="-gpu" --cpu_per_rs 1 --tasks_per_rs 1 --gpu_per_rs 1 --nrs`
       is ok but
       `jsrun  --nrs 2 --smpiargs="-gpu" --cpu_per_rs 1 --tasks_per_rs 1 --gpu_per_rs 1`
       or
       `jsrun --smpiargs="-gpu" --cpu_per_rs 1 --tasks_per_rs 1 --gpu_per_rs 1 --nrs 2`
       will fail.
     - `--runCholla=false` - Makes the system test not launch Cholla. Generally this and `--compareSystemTestResults=false` should only be used for
      large MPI tests where the user wishes to separate the execution of cholla
       and the comparison of results onto different machines/jobs.
     - `--compareSystemTestResults=false` - Launches Cholla, but does not compare the results to fiducial data.
     - Any GoogleTest flags you might desire. Noteworthy flags: `--gtest_filter`
       can be used to only run certain tests and `--gtest_output` can be use to
       generate XML or JSON reports

### GoogleTest Command Line Arguments

A full list of GoogleTest's command line arguments and their usage can be found in their [Running Tests: Advanced Options](https://google.github.io/googletest/advanced.html#running-test-programs-advanced-options) documentation or by running a GoogleTest executable with the `--help` argument. Here are a few options of particular note, please refer to the GoogleTest documentation for full details

- `--gtest_list_tests` - Lists names of all tests, does not run them
- `--gtest_filter=POSITIVE_PATTERNS[-NEGATIVE_PATTERNS]` - Filters tests so that only those that match the positive patterns and don't match the negative patterns are run. Note that the patterns are case sensitive and multiple positive patterns use an inclusive or, all tests that match any positive pattern will be run.
- `--gtest_output=(json|xml)[:DIRECTORY_PATH/|:FILE_PATH]` - generate a report on the results of the tests
- `--gtest_death_test_style=(fast|threadsafe)` - Choose death test style. Set to `threadsafe` within our testing `main` function. Don't change this unless you know what you're doing.
- There are a series of flags that effect how GTest handles failures. Includes setting debugger break points, C++ exceptions, etc.

## Writing New Tests

In general any test consists of a couple of components

- A call to one of GoogleTest's `TEST` macros. Usually this will just be the
  [`TEST`](https://google.github.io/googletest/reference/testing.html#TEST)
  macro but there are other options for
  [fixtures](https://google.github.io/googletest/reference/testing.html#TEST_F)
  and
  [parameterized](https://google.github.io/googletest/reference/testing.html#TEST_P)
  tests.
- All the different `TEST` macros have two arguments, the test suite/fixture
  name and the individual test name. The full name of the test is the
  suite/fixture name followed by the test name seperated by a period. Details of
  our naming conventions can be found in the [Naming Scheme](#naming-scheme)
  section.
  - Example `testSuiteName.IndividualTestName`
- Test suites are groups of tests that are logically connected, for example all
  the tests for the HLLC Riemann solver might be in a single test suite.
  [Test fixtures](https://google.github.io/googletest/primer.html#same-data-multiple-tests)
  are also used for logically grouping tests but in addition they allow custom
  set up and tear down features that can be shared between test. Each test will
  use its own instance of the fixture.
  - Example: If you're testing the methods of the `Grid3D` class and you want to
    initialize the `Grid3D` object the same way for many different tests you can
    use a fixture and then the setup would automatically be done for each test
    in that fixture. Note that each test would use a new instance of the
    `Grid3D` class in this example. Resources can be shared between tests but
    the details are beyond the scope of this guide, please refer to GoogleTest's
    documentation
- After you've setup the `TEST` macro then you can run any C++ code you need for
  your test and use GoogleTest
  [assertions](https://google.github.io/googletest/primer.html#assertions) to
  check the results. A full list of assertions can be found in GoogleTest's docs
  under [Assertions Reference](https://google.github.io/googletest/reference/assertions.html).
- All assertions have two different versions, `ASSERT_*` and `EXPECT_*`. The
  `ASSERT` versions test the result and *stop the current test if it fails*
  whereas the `EXPECT` version tests the results and allows the test to continue
  running if it fails. Generally you should use the `EXPECT` version, so that a
  test failure gives you as much information as possible unless there's a reason
  that you must stop the test if something fails.
  - Example: If you're looping through test results from an HLLC test, you want
    to know all the fluxes that are wrong so you would use `EXPECT`. If
    you're checking that two datasets in HDF5 files have the same names before
    you compare their contents you would use an '`ASSERT`, since there's no point
    in comparing energy to density.
- Example:

      ```cpp
      // External Libraries and Headers
      #include <gtest/gtest.h>

      // Local includes
      #include "../path_to_header_of_function_thats_being_tested"

      TEST(testSuiteName,
           individualTestName)
      {
          // run function to be tested
          EXPECT_EQ(fiducialResult, testResult);
      }
      ```

### Comparing Floating Point Numbers

Floating point comparisons are notoriously tricky and GoogleTest's default
floating point comparison tools are to restrictive for our needs. As such we've
written a more sophisticated tool for comparing double precision (FP64) numbers
in the `testingUtilities` namespace declared in
{repository-file}`src/utils/testing_utilities.h`.
There's two relevant functions there, `ulpsDistanceDbl` which computes the
[ULP distance](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
between two FP64 numbers and the `nearlyEqualDbl` function which checks if two
FP64 numbers are within an absolute margin or a ULP margin. Details can be found
in the doxygen documentation in
{repository-file}`src/utils/testing_utilities.h` and a usage
example can be found at
{repository-file}`src/riemann_solvers/hllc_cuda_test.cu`.
If required in the future FP32, or any other precision, versions of
`nearlyEqualDbl` and `ulpsDistanceDbl` can be added easily.

### Writing New Unit or Integration Tests

New unit or integration tests, or any non-system test, is done using standard
GoogleTest API calls and should be named and located according to the guidelines
in [Naming Scheme](#naming-scheme). Details on the functionality of GoogleTest
can be found in the [GoogleTest Docs](https://google.github.io/googletest/primer.html).
There are many excellent guides on what should be tested, how to test it, and
how to write quality tests.
Some of those resources can be found in the {ref}`testing-useful-links-guides` section.
Below are some basics to get you started, and an example test can be found in
{repository-file}`src/riemann_solvers/hllc_cuda_test.cu`:

### Testing Strategies

- What should you test exactly?
  - Test each path within the code. I.e. each branch in every `if-elif-else`
    statement or any control branching
  - Test with good inputs *and* bad inputs, the code should fail when given bad
    inputs and it should fail the way you expect it to.
  - Test edge cases. Things that are uncommon but could throw a wrench in
    things
- Death Tests - test that the code crashes, segfaults, etc when it should.
  - Special naming convention should be used. Specifically the name of all test
    suites/fixtures with a death test in them should end in `DeathTest`
    ([Reason](http://google.github.io/googletest/advanced.html#death-tests-and-threads))
- Performance Tests
  1. Benchmark the code. There's a library for this or you can do it yourself
  2. Compare the execution time to the fiducial time with EXPECT_NEAR using an
     appropriate margin of error

### Writing New System Tests

System tests run the entirety of the code from start to finish with a specific
problem. Examples include Sod Shock Tubes, convergence tests, etc. Problems with analytical solutions are useful as
they as they provide a known correct result.

Writing a system test from scratch can be done using the class
(`systemTest::SystemTestRunner`) that does almost all the work for you. There
are two basic ways to run a system test: with the fiducial data stored in an
HDF5 file or with the fiducial data generated by the program on the fly. The
first is useful for general regression testing, whereas generating data on the fly might be
suitable for test cases with simple, analytical results or parameterized tests.
The `systemTest::SystemTestRunner.runTest()` method runs Cholla, checks
the results, and stores the output in `cholla/bin/TestSuiteName_TestName/` for
later examination if needed.

#### System Test with Fiducial HDF5 Files

An example of a system test with a fiducial HDF5 file can be found
in
{repository-file}`src/system_tests/hydro_system_tests.cpp`.
Here are detailed instructions:

1. If it doesn't already exist, create a file called `makeType_system_tests.cpp`
   in `cholla/src/system_tests` where `makeType` is the make type of that system
   test (e.g. hydro, particle, gravity, etc).
2. Include both GoogleTest and the {repository-file}`src/system_tests/system_tester.h`
   header.
3. Name your new system test according to the conventions in the
   [Naming Scheme](#naming-scheme) section. Instantiate a
   `systemTest::SystemTestRunner` object, then call the `runTest` method. Naming
   the test according to convention is very important since the
   `systemTest::SystemTestRunner` class uses the test name to find the settings
   and fiducial data files and to determine where to put the output files.
   - Example:
      ```cpp
      // External Libraries and Headers
      #include <gtest/gtest.h>

      // Local includes
      #include "../system_tests/system_tester.h"

      TEST(tHYDROSYSTEMSodShockTube,
           CorrectInputExpectCorrectOutput)
      {
        systemTest::SystemTestRunner sodTest;
        sodTest.runTest();
      }
      ```
4. Store the input file used in `cholla/src/system_tests/input_files` and name
   it `TestSuiteName_TestName.txt`.
   - Example: for the test above the input file should be named
     `tHYDROSYSTEMSodShockTube_CorrectInputExpectCorrectOutput.txt`
5. Store the fiducial data file that is being compared against in
   the `system_tests` directory of the
   [cholla-tests-data repository](https://github.com/cholla-hydro/cholla-tests-data)
   and name it `TestSuiteName_TestName.h5`. Note that this will require Git LFS
   to be installed on your system
   - Note: You must use HDF5 files that are formatted the same way that Cholla
     formats HDF5 files. Text files will not work.
   - Note: MPI system tests still expect a single fiducial file that encompasses
     the entire domain.
   - Example: for the test above the fiducial data file should be named
     `tHYDROSYSTEMSodShockTube_CorrectInputExpectCorrectOutput.h5`

#### System Tests for Periodic Problems (waves)

If the problem you want to test is a wave, or similar periodic problem, then
instead of using a fiducial data file you can check for correctness by computing
the L1 error of the final state vs. the initial state. This can be done easily
with the `runL1ErrorTest` method instead of the `runTest` method and doesn't
require a fiducial data file. Other than that all steps are similar to a system
tests with fiducial data file, see the doxygen documentation in the header file
for more info.

```cpp
// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

TEST(tHYDROSYSTEMSoundWave,
      CorrectInputExpectCorrectOutput)
{
  systemTest::SystemTestRunner soundwaveTest;
  soundwaveTest.runL1Error(maxAllowedErrorHere);
}
```

#### System Tests with Particles

Since the particle data is stored in a different file and different format than
the hydro fields data slightly more work is required to get it working. The
`SystemTestRunner` class requires that all the fiducial data be stored in a
single HDF5 file. To combine the fields file (e.g. `1.h5.0`) and the particle
file (e.g. `1_particles.h5.0`) use the `combine_hydro_particles.py` script in
`python_sripts`. Simple edit the path variables to point at your file and then
run the script then put the combined file in the `system_tests` directory of the
[cholla-tests-data repository](https://github.com/cholla-hydro/cholla-tests-data)
with the usual naming convention. If your test case is particles only and does
not output a hydro field file then all you need to do is change the name of the
`density` dataset in the `_particles.h5` to `particle_density` then name and
locate the file according to convention.

As well as setting up the fiducial files correctly you must set the constructor
to tell the object to look for particle data files (or not to look for hydro
data files). If you just want to turn on particle data then you can initialize
the object like

```cpp
// Turn on particle data
systemTest::SystemTestRunner particleAndHydro(true);

// Turn on particle data and turn off hydro data
systemTest::SystemTestRunner particleOnly(true, false);
```

#### Advanced System Tests

If you want to generate the fiducial data on the fly or have finer control over
the system test then the `systemTest::SystemTestRunner` class has a selection of
methods to help you do that. Full details for each method are found within
doxygen comments in the {repository-file}`src/system_tests/system_tester.h` header.

- `Constructor`: When initializing the object make sure to pass the appropriate
  booleans to indicate if you're using particle data, hydro data, a fiducial
  data file and/or settings file. The default is to use hydro data, not use
  particle data, and use both a fiducial data file and a input file
- `runTest`: Runs the system test as set up
- `chollaLaunchParams`: A `std::string` of Cholla settings overrides to
  use when launching Cholla. Details in {gh-pr}`87`.
- `numMpiRanks`: A member variable that indicates how many MPI ranks of Cholla
  to run, defaults to 1.
- Various getters for path variables, all return std::string objects:
  - `getChollaPath` - The path to the Cholla executable
  - `getChollaSettingsFilePath` - The full filename/path of the settings file
    used to launch Cholla
  - `getOutputDirectory` - The path to the directory where output is stored
  - `getConsoleOutputPath` - The full filename/path to the file where all the
    console output is stored.
- `getDataSetsToTest`: Return a `std::vector<std::string>` of all the dataset
  names that will be compared. If an fiducial HDF5 file is used it defaults to a
  vector of all the datasets. If no fiducial HDF5 is used then it defaults to
  empty and must be set manually with `setDataSetsToTest`.
- `setDataSetsToTest`: Choose which data sets to compare. Overwrites if there
  are already names in the vector. Must be called if no HDF5 file is provided
- `getFiducialFile`: Returns the `H5::H5File` object for the fiducial HDF5 file
- `getTestFile`: Returns the `H5::H5File` object for the test HDF5 file
- `setCompareNumTimeSteps`: Choose whether or not to compare the number of time
  steps
- `generateConstantData`: Takes a double value and shape of an field then
  returns a vector populated by that value. Note that it's always a 1D vector
  whose size is the product of the shape requested. So if you ask for a 10x10x10
  vector it returns a 1D vector with 1,000 elements. Can be indexed with this
  equation `index = (xIndex * lengthInY + yIndex) * lengthInZ + zIndex;`
- `generateSineData`: Takes a series of values and a shape then returns an
  vector containing a sine wave with those settings. Also returns a 1D vector.
- `setFiducialData`: Used to add a new fiducial dataset to be compared against.
  Takes the field name and vector
- `setFiducialNumTimeSteps`: Set the number of fiducial time steps. Only works
  if no fiducial file is used.
- The `globalRunCholla` and `globalCompareSystemTestResults` global variables
  can be used within your system test to control which parts are run in the case
  of a run where either Cholla is not run or the comparison is not performed.
  These global variables are set by the `-runCholla=false` and
  `--compareSystemTestResults=false` command line arguments and should only be
  used for large MPI tests where the user wishes to separate the execution of
  cholla and the comparison of results onto different machines/jobs. Note that
  the comparison is always done single threaded and so even if the number of
  ranks is large the overall domain should be small, on the order of 10^3 cells
  per GPU.
- Examples:

    ```cpp
    // Example for generating fiducial data on the fly rather than getting it
    // from a fiducial data file

    // External Libraries and Headers
    #include <gtest/gtest.h>
    #include <vector>
    #include <string>

    // Local includes
    #include "../system_tests/system_tester.h"
    TEST(testSuiteName,
         individualTestName)
    {
      // Instantiate the object
      systemTest::SystemTestRunner testObject;

      // Choose two fields to compare
      std::vector<std::string> fieldNames{"newField1", "newField2"};
      testObject.setDataSetsToTest(fieldNames);

      // Don't compare the time steps
      testObject.setCompareNumTimeSteps(false);

      // Generate a constant vector and add them as fiducial data to compare against
      testObject.setFiducialData(fieldName[0], // The name of this dataset
                                 testObject.generateConstantData(1.2, 10, 10, 10))
      testObject.setFiducialData(fieldName[0], // The name of this dataset
                                 testObject.generateConstantData(3.1, 10, 10, 10))

      // Run the test
      testObject.runTest();
    }
    ```

    ```cpp
    // Example for using the class to just launch Cholla then checking for
    // correctness yourself

    // External Libraries and Headers
    #include <gtest/gtest.h>
    #include <vector>
    #include <string>

    // Local includes
    #include "../system_tests/system_tester.h"
    TEST(testSuiteName,
         individualTestName)
    {
      // Instantiate the object
      systemTest::SystemTestRunner testObject(false, false, false);

      // Launch Cholla
      testObject.launchCholla();

      // Get the various paths you might need
      std::string chollaPath        = testObject.getChollaPath();
      std::string settingFilePath   = testObject.getChollaSettingsFilePath();
      std::string outputDirectory   = testObject.getOutputDirectory();
      std::string consoleOutputFile = testObject.getConsoleOutputPath();

      // Check for correctness and capture results
      /* your code here */

      // Run GoogleTest assertion
      EXPECT_TRUE(result);
    }
    ```

## Naming Scheme

GoogleTest allows you to choose which tests to run at runtime via standard
pattern matching. To facilitate this we need to have rigorous system for naming
test suites/fixtures and tests. The build system and the
`systemTest::systemTestRunner`
function both require consistent naming of files as well to function properly.

All tests have a suite/fixture name and the individual test name. The full name
of the test is the suite/fixture name followed by the test name seperated by a
period. All test suite/fixture and individual test names must be must be valid
C++ identifies and shouldn’t include underscores. i.e. you can use A-Z, a-z,
0-9, and cannot start with a number. The 'no underscores' rule is a loose
requirement in GoogleTest as underscores in various configurations are reserved
by both the C++ standard and GoogleTest itself. While the rule is overly strict
it avoides myriad special cases. Details can be found
[here](https://github.com/google/googletest/blob/master/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore).

Here are some examples of full test names to refer to:

- Example unit test name for the HLLC Riemann solver
  `tHYDROtGRAVITYCalculateHLLCFluxesCUDA.LeftSideExpectCorrectOutput`
- System test `tHYDROSYSTEMSodShockTube.CorrectInputExpectCorrectOutput`

### File Naming Conventions

- All the tests for the code units in a specific file should be in a test file
  that sits right next to the file being tested. Test file names should be the
  name of the file being tested with a `_tests` at the end. This convention is
  used in the build system and failing to follow it will likely result in
  compile time linking errors.
  - Example: the tests file for `hllc_cuda.cu` is `hllc_cuda_tests.cu`
- System tests for a given make type should all be in the
  `TypeName_system_tests.cpp` file within `cholla/src/system_tests`.
  - Example: All the system tests for the `hydro` make type should be in the
    {repository-file}`src/system_tests/hydro_system_tests.cpp`
    file.
- System test settings and fiducial data files should be named according to the
  requirements in the [Writing New System Tests](#writing-new-system-tests)
  section.

### Test Suite/Fixture Naming Conventions

- All test suite/fixture names need to start with an list of the make types
  for that suite/fixture in all capital letters; if the test suite/fixture is for all
  make types then use the `ALL` keyword. To facilitate easy pattern matching at
  runtime each make type should start with a lowercase `t` so that we can match patterns
  like `tHYDRO` rather than just `HYDRO` which could show up in another part of the
  full test name
  - Example: `tHYDROtPARTICLEStGRAVITY` would be the first part of a
    suite/fixture name that only runs for the hydro, particles, and gravity
    make types. `tALL` would run the test during all make types.
- Test suite/fixture names should be in PascalCase and clearly indicate what code unit (class, method,
  function, etc) is being tested and if there are multiple suites/fixtures for a code
  unit then the test name should clearly indicate what makes a specific suite
  unique. If the code unit name contains underscores then omit them and write the name in
  PascalCase unless that clearly doesn't make sense, i.e. acronyms and similar.
  - Example: `tHYDROCalculateHLLCFluxesCUDA`
- The name of any test suite/fixture which contains a death test must end in
  `DeathTest`. This is a special flag that GoogleTest uses to help aid in
  thread safety.
  - Example: `tHYDROCalculateHLLCFluxesCUDADeathTest`
- System test suite/fixture names must have the list of make types followed by the
  `SYSTEM` keyword then the problem under test.
  - Example: `tHYDROSYSTEMSodShockTube`

### Test Naming Conventions

- Test names should be in PascalCase and clearly indicate what specific state is
  being tested, the word `Expect`, then what the desired result is. The general
  format should be `StateUnderTestExpectBehavior`.
  - Examples
    - `SodShockLeftSideExpectCorrectOutput`
    - `SodShockLeftSideExpectSegFault`
- System test names should follow the rules for test names. If it's an MPI test
  then the number of ranks should be noted after the state being tested followed
  by `MpiRanks`; if the number of ranks is 1,000 or over then a `k` can be used.
  The phrase `Mpi` should not be used in tests that do not run MPI
  - Examples:
    - `CorrectInputExpectCorrectOutput`
    - `CorrectInput8MpiRanksExpectCorrectOutput`
    - `CorrectInput50kMpiRanksExpectCorrectOutput`
    - `NegativePressureExpectSegfault`

## Automated Testing

All the NIVIDIA builds are build and testing on Pitt CRC hardware using Jenkins on each pull request. Additionally, as part of the Jenkins pipeline, clang-tidy is run on each build type.

To test the AMD builds a GitHub Actions matrix job has been setup in
{repository-file}`.github/workflows/build_tests.yml`
that builds every make type of Cholla against HIP+Clang. Note that
this GitHub Actions workflow *does not* actually run the tests, it just builds
Cholla and the tests to make sure that everything compiles.

## Known issues

### Summit

- Since the system tests launch MPI jobs the tests must be run from within a job,
  running on the header nodes is blocked by Summit
- Currently running the tests when compiled with GCC doesn't work on Summit, you
  have to use XL. The cause is unclear since GCC works fine on other systems

## Glossary

The glossary below provides the meaning of various terms used on this page:

- "Unit test" - Test a single "unit" of code such as a function, class method,
  constructor, etc.
  - Example: Testing that a function that just computes pressure
- "Integration Test" - Test how multiple units of code work together.
  - Example: testing an entire Riemann solver instead of just the individual
    functions within it
- "System Test" or "End-to-End (E2E) Test" - Testing the entire code to make
  sure it produces the correct result or action. In our case this can be done
  with the `systemTest::systemTestRunner` function
- "Regression test" - Test that new changes have not broken the code or
  introduced new bugs. Unit, integration, and system tests can all be regression
  tests
- "Performance Test" - Make sure the thing you're testing doesn't take more time,
  memory, or other resources than expected
- "Mocking" - Some tests might have a dependency that doesn't behave in a
  predictable way for testing or is otherwise undesirable to run for testing and
  so we need to make fake or "mocked" version that does.
  - Example: A random number generator. In production we want it to be random
    but in testing we might always want it to return the same value so that a
    code unit that depends on it produces deterministic results
- "Automated Testing" - Running tests automatically based on some event.
  Typically this event is something like pushing to GitHub or opening a pull
  request.
- "Continuous Integration (CI)" - Quickly integrating changes from different
  developers using some sort of automated build, test, deployment system. See
  the links in the Useful Links & Guides section for more info.
- "Test Driven Development (TDD)" - Writing the tests *before* you write the code
  that will be tested. This can be done as part of the design phase to make sure
  the end product matches all the requirements.

(testing-useful-links-guides)=
## Useful Links & Guides

- *Note that a lot of these links lead to specific companies who are selling
  testing software or something similar. The guides are still good, they're just
  trying to promote their own product. I'm not endorsing their product or
  suggesting we use it, I just liked the guide.*
- *links marked with a '1' are must reads, '2' means helpful but not required,
  '3' is for additional reading*
- (1)[Software Testing Methodologies](https://smartbear.com/learn/automated-testing/software-testing-methodologies/)
- (1)[GoogleTest Primer](http://google.github.io/googletest/primer.html)
- (2)[Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
  - This appears to be the bible on comparing floating point numbers. Most other
    articles I found linked back to this as their primary source and GoogleTest
    cites it as their rational for their FP comparison assertions. This is also
    part of an excellent series of blogs on floating point numbers
- (2)[What is Continuous Integration?](https://www.atlassian.com/continuous-delivery/continuous-integration)
- (2)[What Is End To End Testing? A Helpful Introductory Guide](https://www.testim.io/blog/end-to-end-testing-guide/)
- (2)[Test Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- (2)[How to test software, part I: mocking, stubbing, and contract testing](https://circleci.com/blog/how-to-test-software-part-i-mocking-stubbing-and-contract-testing/)
- (3)Best Practices for HPC Software Developers: #4 "Testing & Documenting Your Code" ([video](https://www.youtube.com/watch?v=kAC0N84JaHA))([slides](https://www.olcf.ornl.gov/wp-content/uploads/2016/04/testing_webinar.pdf))
- (3)[Automated Regression Testing: Everything You Need To Know](https://www.testim.io/blog/automated-regression-testing/)
- (3)[Software Testing](https://en.wikipedia.org/wiki/Software_testing#Testing_approach)
- (3)[HPC testing best practices, one of the talks here](https://ideas-productivity.org/events/hpc-best-practices-webinars/)
- (3)[gtest MPI listener](https://github.com/LLNL/gtest-mpi-listener)

