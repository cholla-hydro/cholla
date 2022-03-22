
/*!
 * \file cuda_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu), Helena Richie (helenarichie@pitt.edu)
 * \brief Tests for the contents of cuda_utilities.h and cuda_utilities.cpp
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/testing_utilities.h"
#include "../utils/cuda_utilities.h"
#include "../global/global.h"

/*
 PCM : n_ghost = 2
 PLMP : n_ghost = 2
 PLMC : n_ghost = 3
 PPMP : n_ghost = 4
 PPMC : n_ghost = 4
*/

// =============================================================================
// Local helper functions
namespace
{
    struct TestParams
    {
        std::vector<int> n_ghost {2, 2, 3, 4};
        std::vector<int> nx {1, 2048, 2048, 2048};
        std::vector<int> ny {1, 2048, 2048, 2048};
        std::vector<int> nz {1, 4096, 4096, 4096};
        std::vector<std::string> names{"Single-cell 3D PCM/PLMP case", "Large 3D PCM/PLMP case", "Large PLMC case", "Large PPMP/PPMC case"};

    };
}

TEST(tHYDROSYSTEMCudaUtilsGetRealIndices, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<std::vector<int>> fiducial_indices {{1, 2, 3, 4, 5, 6}, 
                                               {4, 5, 6, 7, 8, 9}, 
                                               {7, 8, 9, 4, 5, 6},
                                               {7, 8, 9, 4, 5, 6}}; 

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        int is;
        int ie;
        int js;
        int je;
        int ks;
        int ke;
        cuda_utilities::Get_Real_Indices(parameters.n_ghost.at(i), parameters.nx.at(i), parameters.ny.at(i), parameters.nz.at(i), is, ie, js, je, ks, ke);

        std::vector<int> test_indices {is, ie, js, je, ks, ke};

        for (int j = 0; j < test_indices.size(); j++) 
        {
            testingUtilities::checkResults(fiducial_indices[i], test_indices[i][j], parameters.names[i]);
        }
    }
}