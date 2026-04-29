/*!
 * \file
 * \brief Tests for the shared constructs
 */

// Standard Library includes
#include <stdio.h>

#include <type_traits>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"
#include "../utils/shared.h"

// define some helper functions

__global__ void check_empty_kernel_(const SharedHandle<long> handle, bool* out_ptr) { *out_ptr = bool(handle); }

__global__ void check_empty_kernel_(const SharedDevPtr<long> ptr, bool* out_ptr) { *out_ptr = bool(ptr); }

template <typename SharedConstructT>
bool device_check_empty_(const SharedConstructT& shared_construct)
{
  bool* dev_ptr;
  GPU_Error_Check(cudaMalloc(&dev_ptr, sizeof(bool)));

  // actually run the command!
  hipLaunchKernelGGL(check_empty_kernel_, dim3(1), dim3(1), 0, 0, shared_construct, dev_ptr);
  GPU_Error_Check();
  GPU_Error_Check(cudaDeviceSynchronize());

  // now, let's copy the result back to the host and return it
  bool out;
  GPU_Error_Check(cudaMemcpy(&out, dev_ptr, sizeof(bool), cudaMemcpyDeviceToHost));
  GPU_Error_Check(cudaFree(dev_ptr));
  return out;
}

__global__ void get_wrapped_kernel_(const SharedHandle<long> handle, long* out_ptr) { *out_ptr = handle.get(); }

__global__ void get_wrapped_kernel_(const SharedDevPtr<long> ptr, long** out_ptr) { *out_ptr = ptr.get(); }

template <typename SharedConstructT>
typename SharedConstructT::wrapped_ref_type device_get_(const SharedConstructT& shared_construct)
{
  using OutT = typename SharedConstructT::wrapped_ref_type;
  OutT* dev_ptr;
  GPU_Error_Check(cudaMalloc(&dev_ptr, sizeof(OutT)));

  // actually run the command!
  hipLaunchKernelGGL(get_wrapped_kernel_, dim3(1), dim3(1), 0, 0, shared_construct, dev_ptr);
  GPU_Error_Check();
  GPU_Error_Check(cudaDeviceSynchronize());

  // now, let's copy the result back to the host and return it
  OutT out;
  GPU_Error_Check(cudaMemcpy(&out, dev_ptr, sizeof(OutT), cudaMemcpyDeviceToHost));
  GPU_Error_Check(cudaFree(dev_ptr));
  return out;
}

// this is used to test common properties of both SharedDevPtr and SharedHandle
template <typename SharedConstructT>
class tALLSharedConstructTest : public testing::Test
{
  long n_deleted_ = 0;
  long n_created_ = 0;

 public:
  SharedConstructT create_nonempty()
  {
    // get the pointer to the deletion_counter
    long* deletion_counter_ptr         = &n_deleted_;
    [[maybe_unused]] long creation_num = n_created_;
    n_created_++;

    if constexpr (std::is_same_v<SharedConstructT, SharedHandle<long>>) {
      // for the purposes of the test, the handle has a unique and arbitrary value
      long handle  = creation_num + 3;
      auto deleter = [deletion_counter_ptr](long& handle) {
        (*deletion_counter_ptr)++;
        // since this is a fake handle that isn't actually associated with a real
        // resource, do nothing
      };
      return {handle, deleter};
    } else if constexpr (std::is_same_v<SharedConstructT, SharedDevPtr<long>>) {
      // allocate the pointer
      long* new_dev_ptr;
      GPU_Error_Check(cudaMalloc(&new_dev_ptr, sizeof(long)));

      // allocate the deleter
      auto deleter = [deletion_counter_ptr](long* dev_ptr) {
        (*deletion_counter_ptr)++;
        GPU_Error_Check(cudaDeviceSynchronize());  // <- this is just to be safe
        GPU_Error_Check(cudaFree(dev_ptr));
      };
      return {new_dev_ptr, deleter};
    } else {
      static_assert(always_false<SharedConstructT>, "received unexpected type");
    }

    CHOLLA_ERROR("SHOULD BE UNREACHABLE");  // <- silences warning about not returning
  }

  int num_deleted() const { return n_deleted_; }
};

using MyTypes = ::testing::Types<SharedHandle<long>, SharedDevPtr<long>>;
TYPED_TEST_SUITE(tALLSharedConstructTest, MyTypes);

TYPED_TEST(tALLSharedConstructTest, CheckEmptyHost)
{
  TypeParam empty_shared_resource;
  EXPECT_FALSE(bool(empty_shared_resource));
}

TYPED_TEST(tALLSharedConstructTest, CheckEmptyDevice)
{
  TypeParam empty_shared_resource;
  EXPECT_FALSE(device_check_empty_(empty_shared_resource));
}

TYPED_TEST(tALLSharedConstructTest, NonEmptyDestructorReleaseOwnership)
{
  {
    TypeParam shared_resource = this->create_nonempty();

    // make sure the construct reports that it's not empty
    EXPECT_TRUE(bool(shared_resource)) << "should report that the construct isn't empty (on the host)";
    EXPECT_TRUE(bool(shared_resource)) << "should report that the construct isn't empty (on the device)";

    // make sure the construct reports the same wrapped value on host and on device
    typename TypeParam::wrapped_ref_type from_device = device_get_(shared_resource);
    EXPECT_EQ(shared_resource.get(), from_device)
        << "the result of the get method should be the same on host & on device";

    // at this point nothing should have been free-ed
    ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";
  }
  ASSERT_EQ(this->num_deleted(), 1) << "the construct should be called (1 time) after the construct goes out of scope";
}

TYPED_TEST(tALLSharedConstructTest, NonEmptyResetReleaseOwnership)
{
  TypeParam shared_resource = this->create_nonempty();

  // make sure the construct reports that it's not empty
  EXPECT_TRUE(bool(shared_resource)) << "should report that the construct isn't empty (on the host)";
  EXPECT_TRUE(bool(shared_resource)) << "should report that the construct isn't empty (on the device)";

  // make sure the construct reports the same wrapped value on host and on device
  typename TypeParam::wrapped_ref_type from_device = device_get_(shared_resource);
  EXPECT_EQ(shared_resource.get(), from_device)
      << "the result of the get method should be the same on host & on device";

  // at this point nothing should have been free-ed
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";

  shared_resource.reset();

  ASSERT_EQ(this->num_deleted(), 1) << "the construct should be called (1 time) after the construct releases "
                                    << "ownership via the reset() method";
}

TYPED_TEST(tALLSharedConstructTest, CopyConstructEmpty)
{
  TypeParam empty_shared_resource;

  // the disabled linter check on the next line is telling us that the code would be
  // faster if we got rid of the instance named copy and perform all checks directly on
  // empty_shared_resource (from a performance stand-point it's completely correct as
  // long as the test that we perform is successful)
  TypeParam copy(empty_shared_resource);  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_FALSE(bool(empty_shared_resource)) << "the original should be empty";
  EXPECT_FALSE(bool(copy)) << "the copy also should be empty";
}

TYPED_TEST(tALLSharedConstructTest, CopyConstructNonEmpty)
{
  TypeParam shared_resource = this->create_nonempty();
  TypeParam copy(shared_resource);

  EXPECT_TRUE(bool(shared_resource)) << "the original shouldn't be empty";
  EXPECT_TRUE(bool(copy)) << "the copy also shouldn't be empty";
  EXPECT_EQ(shared_resource.get(), copy.get()) << "the original and copy should both share ownership";

  // validate that the deleter is only called after there are no remaining owners
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";
  copy.reset();
  ASSERT_EQ(this->num_deleted(), 0) << "Nothing should be freed while there's still an owner of the";
  shared_resource.reset();
  ASSERT_EQ(this->num_deleted(), 1) << "Memory leak";
}

TYPED_TEST(tALLSharedConstructTest, MoveConstructEmpty)
{
  TypeParam empty_shared_resource;
  TypeParam empty_shared_resource2(std::move(empty_shared_resource));

  // the next line is validating behavior in the exact circumstance that linting seeks to avoid
  // NOLINTNEXTLINE(bugprone-use-after-move, hicpp-invalid-access-moved)
  EXPECT_FALSE(bool(empty_shared_resource)) << "the original should be empty";

  // check variable where value was moved to
  EXPECT_FALSE(bool(empty_shared_resource2)) << "the move-constructed instance also should be empty";
}

TYPED_TEST(tALLSharedConstructTest, MoveConstructNonEmpty)
{
  TypeParam shared_resource = this->create_nonempty();
  TypeParam shared_resource2(std::move(shared_resource));

  // the next line is validating behavior in the exact circumstance that linting seeks to avoid
  // NOLINTNEXTLINE(bugprone-use-after-move, hicpp-invalid-access-moved)
  EXPECT_FALSE(bool(shared_resource)) << "original should be empty";

  // check variable where value was moved to
  EXPECT_TRUE(bool(shared_resource2)) << "move constructed instance shouldn't be empty";

  // validate that the deleter is only called after there are no remaining owners
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";
  shared_resource2.reset();
  ASSERT_EQ(this->num_deleted(), 1) << "Memory leak";
}

TYPED_TEST(tALLSharedConstructTest, SelfCopyAssign)
{
  // this may seem silly, but this is a corner case we should explicitly test
  // (for context, doing a self-move-assignment usually just isn't done)
  TypeParam shared_resource = this->create_nonempty();
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";

  // do the self assign.
  //
  // the disabled lint tries to prevent this operation because a self-assign shouldn't
  // change anything about the instance being self-assigned (i.e. this is the invariant
  // this test is actually checking) and would just waste CPU cycles
  shared_resource = shared_resource;  // NOLINT(misc-redundant-expression)
  ASSERT_EQ(this->num_deleted(), 0) << "the internal reference count erroneously hit 0 during the operation";

  // now confirm that releasing the resource creates the appropriate result
  shared_resource.reset();
  ASSERT_EQ(this->num_deleted(), 1) << "Memory leak";
}

TYPED_TEST(tALLSharedConstructTest, CopyAssignEmptyToEmpty)
{
  TypeParam empty_1;
  TypeParam empty_2;

  // perform copy assignment
  empty_2 = empty_1;

  EXPECT_FALSE(bool(empty_1));
  EXPECT_FALSE(bool(empty_2));
}

TYPED_TEST(tALLSharedConstructTest, CopyAssignNonEmptyToNonEmpty)
{
  TypeParam shared_resource_1 = this->create_nonempty();
  TypeParam shared_resource_2 = this->create_nonempty();

  EXPECT_NE(shared_resource_1.get(), shared_resource_2.get()) << "SANITY CHECK!";
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";

  typename TypeParam::wrapped_ref_type pre_assign_wrapped_1 = shared_resource_1.get();

  // perform copy assignment
  shared_resource_2 = shared_resource_1;

  ASSERT_EQ(this->num_deleted(), 1) << "The deleter should have been invoke for the resource previously tracked by "
                                    << "shared_resource2";

  EXPECT_TRUE(bool(shared_resource_1)) << "shared_resource_1 shouldn't be empty";
  EXPECT_TRUE(bool(shared_resource_2)) << "shared_resource_2 shouldn't be empty";
  EXPECT_EQ(shared_resource_1.get(), pre_assign_wrapped_1)
      << "the value wrapped by shared_resource_1 should not have changed";
  EXPECT_EQ(shared_resource_1.get(), shared_resource_2.get())
      << "shared_resource_1 and shared_resource_2 should wrap the same value";

  // validate that the deleter is only called after there are no remaining owners
  shared_resource_1.reset();
  ASSERT_EQ(this->num_deleted(), 1) << "Nothing should be freed while there's still an owner of the";
  shared_resource_2.reset();
  ASSERT_EQ(this->num_deleted(), 2) << "Memory leak";
}

TYPED_TEST(tALLSharedConstructTest, MoveAssignEmptyToEmpty)
{
  TypeParam empty_1;
  TypeParam empty_2;

  // perform move assignment
  empty_2 = std::move(empty_1);

  // the next line is validating behavior in the exact circumstance that linting seeks to avoid
  // NOLINTNEXTLINE(bugprone-use-after-move, hicpp-invalid-access-moved)
  EXPECT_FALSE(bool(empty_1));
  EXPECT_FALSE(bool(empty_2));
}

TYPED_TEST(tALLSharedConstructTest, MoveAssignNonEmpty)
{
  TypeParam shared_resource_1 = this->create_nonempty();
  TypeParam shared_resource_2 = this->create_nonempty();

  EXPECT_NE(shared_resource_1.get(), shared_resource_2.get()) << "SANITY CHECK!";
  ASSERT_EQ(this->num_deleted(), 0) << "SANITY CHECK";

  typename TypeParam::wrapped_ref_type pre_assign_wrapped_1 = shared_resource_1.get();

  // perform move assignment
  shared_resource_2 = std::move(shared_resource_1);

  ASSERT_EQ(this->num_deleted(), 1) << "The deleter should have been invoke for the resource previously tracked by "
                                    << "shared_resource2";

  // the next line is validating behavior in the exact circumstance that linting seeks to avoid
  // NOLINTNEXTLINE(bugprone-use-after-move, hicpp-invalid-access-moved)
  EXPECT_FALSE(bool(shared_resource_1)) << "shared_resource_1 should be empty";

  // check variable where value was moved to
  EXPECT_TRUE(bool(shared_resource_2)) << "shared_resource_2 shouldn't be empty";
  EXPECT_EQ(shared_resource_2.get(), pre_assign_wrapped_1)
      << "move assignment should transfer ownership from shared_resource_1 to "
      << "shared_resource_2";

  // validate that the deleter is only called after there are no remaining owners
  shared_resource_2.reset();
  ASSERT_EQ(this->num_deleted(), 2) << "Memory leak";
}