/*!
 * \file device_vector_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the DeviceVector class
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../utils/testing_utilities.h"
#include "../utils/DeviceVector.h"


namespace // Anonymous namespace
{
    template <typename T>
    void checkPointerAttributes(cuda_utilities::DeviceVector<T> &devVector)
    {
        // Get the pointer information
        cudaPointerAttributes ptrAttributes;
        CudaSafeCall(cudaPointerGetAttributes(&ptrAttributes, devVector.data()));

        // Warning strings
        std::string typeMessage          = "ptrAttributes.type should be 2 since "
                                           "that indicates type cudaMemoryTypeDevice. "
                                           "0 is cudaMemoryTypeUnregistered, "
                                           "1 is cudaMemoryTypeHost, and "
                                           "3 is cudaMemoryTypeManaged";
        std::string const deviceMessage  = "The pointer should be on device 0";
        std::string const devPtrMessage  = "The device pointer is nullptr";
        std::string const hostPtrMessage = "The host pointer is not nullptr";

        // Check that the pointer information is correct
        #ifdef  O_HIP
            typeMessage = "ptrAttributes.memoryType should be 1 since that indicates a HIP device pointer.";
            EXPECT_EQ(1, ptrAttributes.memoryType)      << typeMessage;
        #else // O_HIP is not defined i.e. we're using CUDA
            EXPECT_EQ(2, ptrAttributes.type)            << typeMessage;
        #endif  // O_HIP
        EXPECT_EQ(0, ptrAttributes.device)              << deviceMessage;
        EXPECT_NE(nullptr, ptrAttributes.devicePointer) << devPtrMessage;
        EXPECT_EQ(nullptr, ptrAttributes.hostPointer)   << hostPtrMessage;
    }
} // Anonymous namespace

// =============================================================================
// Tests for expected behavior
// =============================================================================
TEST(tALLDeviceVectorConstructor,
     CheckConstructorDataAndSizeExpectProperAllocationAndValues)
{
    // Initialize the DeviceVector
    size_t const vectorSize = 10;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};

    // Check that the size is correct
    EXPECT_EQ(vectorSize, devVector.size());

    // Check the pointer information
    checkPointerAttributes<double>(devVector);
}

TEST(tALLDeviceVectorDestructor,
     CheckDestructorExpectProperDeallocation)
{
   // Initialize the DeviceVector
   size_t const vectorSize = 10;
   cuda_utilities::DeviceVector<double> devVector{vectorSize};

    // Destruct the object
    devVector.~DeviceVector();

   // Get the pointer information
   cudaPointerAttributes ptrAttributes;
   CudaSafeCall(cudaPointerGetAttributes(&ptrAttributes, devVector.data()));

    // Warning strings
    std::string typeMessage          = "ptrAttributes.type should be 0 since "
                                       "that indicates type cudaMemoryTypeUnregistered"
                                       "0 is cudaMemoryTypeUnregistered, "
                                       "1 is cudaMemoryTypeHost, "
                                       "2 is cudaMemoryTypeDevice, and"
                                       "3 is cudaMemoryTypeManaged";
    std::string deviceMessage        = "The pointer should be null which is device -2";
    std::string const devPtrMessage  = "The device pointer is nullptr";
    std::string const hostPtrMessage = "The host pointer is not nullptr";

    // Check that the pointer information is correct
    #ifdef  O_HIP
        typeMessage = "ptrAttributes.memoryType should be 1 since that indicates a HIP device pointer.";
        deviceMessage = "The pointer should be 0";
        EXPECT_EQ(0, ptrAttributes.memoryType)      << typeMessage;
        EXPECT_EQ(0, ptrAttributes.device)          << deviceMessage;
    #else // O_HIP is not defined i.e. we're using CUDA
        EXPECT_EQ(0, ptrAttributes.type)            << typeMessage;
        EXPECT_EQ(-2, ptrAttributes.device)         << deviceMessage;
    #endif  // O_HIP
    EXPECT_EQ(nullptr, ptrAttributes.devicePointer) << devPtrMessage;
    EXPECT_EQ(nullptr, ptrAttributes.hostPointer)   << hostPtrMessage;
}

TEST(tALLDeviceVectorStdVectorHostToDeviceCopyAndIndexing,
     CheckDeviceMemoryValuesAndIndexingOperationsExpectCorrectMemoryValues)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};
    std::vector<double> stdVec(vectorSize);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    // Copy the value to the device memory
    devVector.cpyHostToDevice(stdVec);

    // Check the values in device memory with both the .at() method and
    // overloaded [] operator
    for (size_t i = 0; i < vectorSize; i++)
    {
        EXPECT_EQ(stdVec.at(i), devVector.at(i));
        EXPECT_EQ(stdVec.at(i), devVector[i]);
    }
}

TEST(tALLDeviceVectorArrayHostToDeviceCopyAndIndexing,
     CheckDeviceMemoryValuesAndIndexingOperationsExpectCorrectMemoryValues)
{
   // Initialize the vectors
   size_t const vectorSize = 10;
   cuda_utilities::DeviceVector<double> devVector{vectorSize};
   std::vector<double> stdVec(vectorSize);
   std::iota(stdVec.begin(), stdVec.end(), 0);

   // Copy the value to the device memory
   devVector.cpyHostToDevice(stdVec.data(), stdVec.size());

   // Check the values in device memory with both the .at() method and
   // overloaded [] operator
   for (size_t i = 0; i < vectorSize; i++)
   {
       EXPECT_EQ(stdVec.at(i), devVector.at(i));
       EXPECT_EQ(stdVec.at(i), devVector[i]);
   }
}

TEST(tALLDeviceVectorArrayAssignmentMethod,
     AssignSingleValuesExpectCorrectMemoryValues)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};

    // Perform assignment
    devVector.assign(13);
    devVector.assign(17,4);

    // Check the values in device memory
    EXPECT_EQ(13, devVector.at(0));
    EXPECT_EQ(17, devVector.at(4));
}

TEST(tALLDeviceVectorStdVectorDeviceToHostCopy,
     CheckHostMemoryValuesExpectCorrectMemoryValues)
{
   // Initialize the vectors
   size_t const vectorSize = 10;
   cuda_utilities::DeviceVector<double> devVector{vectorSize};
   std::vector<double> stdVec(vectorSize), hostVec(vectorSize);
   std::iota(stdVec.begin(), stdVec.end(), 0);

   // Copy the value to the device memory
   devVector.cpyHostToDevice(stdVec);

   // Copy the values to the host memory
   devVector.cpyDeviceToHost(hostVec);

   // Check the values
   for (size_t i = 0; i < vectorSize; i++)
   {
       EXPECT_EQ(stdVec.at(i), hostVec.at(i));
   }
}

TEST(tALLDeviceVectorArrayDeviceToHostCopy,
     CheckHostMemoryValuesExpectCorrectMemoryValues)
{
  // Initialize the vectors
  size_t const vectorSize = 10;
  cuda_utilities::DeviceVector<double> devVector{vectorSize};
  std::vector<double> stdVec(vectorSize), hostVec(vectorSize);
  std::iota(stdVec.begin(), stdVec.end(), 0);

  // Copy the value to the device memory
  devVector.cpyHostToDevice(stdVec.data(), stdVec.size());

  // Copy the values to the host memory
  devVector.cpyDeviceToHost(hostVec.data(), hostVec.size());

  // Check the values
  for (size_t i = 0; i < vectorSize; i++)
  {
      EXPECT_EQ(stdVec.at(i), hostVec.at(i));
  }
}

TEST(tALLDeviceVectorReset,
     SetNewSizeExpectCorrectSize)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    size_t const newSize    = 20;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};
    std::vector<double> stdVec(vectorSize), newVec(newSize);
    std::iota(stdVec.begin(), stdVec.end(), 0);
    std::iota(newVec.begin(), newVec.end(), 20);

    // Copy the value to the device memory
    devVector.cpyHostToDevice(stdVec);

    // Reset the vector
    devVector.reset(newSize);

    // Check the size
    EXPECT_EQ(newSize, devVector.size());

    // Check the pointer
    checkPointerAttributes(devVector);

    // Copy the new values into device memory
    devVector.cpyHostToDevice(newVec);

    // Check the values
    for (size_t i = 0; i < newSize; i++)
    {
        EXPECT_EQ(newVec.at(i), devVector.at(i));
    }
}

TEST(tALLDeviceVectorResize,
     SetLargerSizeExpectCorrectSize)
{
    // Initialize the vectors
    size_t const originalSize = 10;
    size_t const newSize    = 20;
    cuda_utilities::DeviceVector<double> devVector{originalSize};
    std::vector<double> stdVec(originalSize);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    // Copy the value to the device memory
    devVector.cpyHostToDevice(stdVec);

    // Reset the vector
    devVector.resize(newSize);

    // Check the size
    EXPECT_EQ(newSize, devVector.size());

    // Check the pointer
    checkPointerAttributes(devVector);

    // Check the values
    for (size_t i = 0; i < originalSize; i++)
    {
        double const fiducialValue = (i < stdVec.size())? stdVec.at(i): 0;
        EXPECT_EQ(fiducialValue, devVector.at(i));
    }
}

TEST(tALLDeviceVectorResize,
     SetSmallerSizeExpectCorrectSize)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    size_t const newSize    = 5;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};
    std::vector<double> stdVec(vectorSize);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    // Copy the value to the device memory
    devVector.cpyHostToDevice(stdVec);

    // Reset the vector
    devVector.resize(newSize);

    // Check the size
    EXPECT_EQ(newSize, devVector.size());

    // Check the pointer
    checkPointerAttributes(devVector);

    // Check the values
    for (size_t i = 0; i < newSize; i++)
    {
        EXPECT_EQ(stdVec.at(i), devVector.at(i));
    }
}

// =============================================================================
// Tests for exceptions
// =============================================================================
TEST(tALLDeviceVectorAt,
     OutOfBoundsAccessExpectThrowOutOfRange)
{
   // Initialize the vectors
   size_t const vectorSize = 10;
   cuda_utilities::DeviceVector<double> devVector{vectorSize};
   std::vector<double> stdVec(vectorSize);
   std::iota(stdVec.begin(), stdVec.end(), 0);

   // Copy the value to the device memory
   devVector.cpyHostToDevice(stdVec);

   // Check that the .at() method throws the correct exception
   EXPECT_THROW(devVector.at(100), std::out_of_range);
}

TEST(tALLDeviceVectorStdVectorHostToDeviceCopy,
    OutOfBoundsCopyExpectThrowOutOfRange)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};
    std::vector<double> stdVec(2*vectorSize);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    // Copy the value to the device memory
    EXPECT_THROW(devVector.cpyHostToDevice(stdVec), std::out_of_range);
}

TEST(tALLDeviceVectorStdVectorDeviceToHostCopy,
    OutOfBoundsCopyExpectThrowOutOfRange)
{
    // Initialize the vectors
    size_t const vectorSize = 10;
    cuda_utilities::DeviceVector<double> devVector{vectorSize};
    std::vector<double> stdVec(vectorSize/2);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    // Copy the value to the device memory
    EXPECT_THROW(devVector.cpyDeviceToHost(stdVec), std::out_of_range);
}
