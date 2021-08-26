/*!
 * \file systemTest.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Header file for the tools to run a system test
 * \date 2021-08-30
 *
 */

#pragma once

/*!
 * \brief The namespace that contains the function for running system tests.
 *
 * \details This namespace contains one function, systemTestRunner, whose
 * purpose is to (as you might expect) run system tests. It's helped in this
 * endeavor with a series of functions in an anonymous namespace in
 * systemTest.cpp that read files, compare results, etc.
 */
namespace systemTest
{
    /*!
     * \brief Runs a system test using the full test name to determine all
     * paths. Pleases note that this function is still in development and does
     * not at this time support MPI runs of cholla or comparing more than one
     * pair of HDF5 files.
     *
     * \details This function uses the full name of your test, i.e. the test
     * suite name plus the test name, along with some global variables to
     * determine the paths to all the input files. The global variables are all
     * set in main_tests.cpp and are the path to the Cholla directory, the make
     * type being used, and the machine being run on. If the main function does
     * get those it will throw an error so that error checking is not done here.
     *
     * To run a system test simply name the test according to convetion and put
     * the input file in the `cholla/src/system_tests/input_files` directory and
     * the data file in the `cholla/src/system_tests/fiducial_data` directory.
     * Then name the files `testSuiteName_testCaseName` with the `.txt` or `.h5`
     * extension respectively. If this function can't find the files it will
     * throw an error with the path it searched. All the output files from the
     * test are deposited in `cholla/bin/testSuiteName_testCaseName`
     *
     */
    void systemTestRunner();
} // namespace systemTest
