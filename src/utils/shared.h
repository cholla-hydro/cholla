/*! \file
 *  declare SharedHandle and SharedDevPtr
 */

#pragma once

#include <atomic>
#include <cstddef>  // std::ptrdiff_t
#include <type_traits>

/*! \defgroup sharedgrp Shared Construct Group
 *
 *  This group describes class templates that provide shared object semantics for some
 *  kind of wrapped value that acts as "a form of reference" to some kind of computing
 *  resource.
 *
 *  At the time of writing, these constructs include:
 *  - \ref SharedHandle wraps a [handle](https://en.wikipedia.org/wiki/Handle_(computing))
 *  - \ref SharedDevPtr wraps a pointer to device memory
 *
 *  These constructs act similarly to a simplified version of std::share_ptr (that also
 *  work on GPUs)
 *
 *  Common Operations
 *  =================
 *  These constructs all provide a common set of operations:
 *
 *  Let's establish the basic semantics modeled by these constructs:
 *  - we say that a construct exercises ownership of an underlying resource by wrapping
 *    a handle or pointer (depending on whether the construct is a \ref SharedHandle or
 *    a \ref SharedDevPtr). The wrapped handle or pointer can be accessed through the
 *    `get()` member function
 *  - a resource can be owned by one or more instance of a construct. When the number of
 *    owners of the resource drops to zero, the construct is deleted.
 *  - an instance of a construct can also be "empty" (i.e. it doesn't own any construct)
 *
 *  Now, let's describe the way that this model is implemented
 *  - the default constructor builds an empty instance
 *  - the primary constructor builds an instance that wraps a previously created handle
 *    or pointer and a function-like callback for deleting the resource.
 *    - The constructed instance takes ownership of the underlying resource referenced
 *      by the handle or pointer.
 *    - This can only happen on the host
 *  - the copy construction and copy assignment are mechanisms for sharing ownership
 *    of an underlying resource. Move construction and move assignment are mechanisms
 *    for transferring ownership of a resource.
 *  - For any kind of assignment (`instance_a = instance_b;` or
 *    `instance_a = std::move(instance_b);`), if `instance_a` and `instance_b` did not
 *    previously share ownership over the same resource, `instance_a` releases
 *    ownership of its previously owned resource in the course of the operation.
 *  - ownership is released with the `reset` method and in an instance's destructor
 *  - the callback function-like deleter object is invoked for a wrapped value (e.g.
 *    handle or pointer) when ownership is released by the last construct that owned
 *    the value
 *
 *  **IMPORTANTLY:** these constructs can be used on the host and on GPUs
 *
 *  How it works:
 *  =============
 *  At a high level, these constructs are implemented using (atomic) reference counting.
 *  Essentially, the constructs hold a pointer to a "control block" that holds a
 *  reference count. When the primary constructor is invoked, the reference count starts
 *  at one. Every drops to 0, the deleter callback is then invoked to delete
 *  the resource.
 *
 *  In slightly more detail, ownership is only tracked on the host. In case its not
 *  obvious why this is a viable strategy, let's make a simple assumption: let's assume
 *  for a moment that we are always extremely careful about releasing device resources
 *  until after all accesses to a resource are complete.
 *
 *  Under that assumption, let's consider the lifetime of a \ref SharedHandle or
 *  a \ref SharedDevPtr a GPU kernel:
 *  - since the primary constructor can only be invoked on the host, the only way to
 *    get an instance of the construct on the device is if we pass it to a kernel
 *    function by value
 *  - within a kernel function no matter how many copies we make of a construct the
 *    number of copies of a construct on the GPU that share ownership of a resource will
 *    drop to zero by time you exit the kernel
 *  - thus, if you imagined counting the total number of instances that share ownership
 *    of a given resource across the CPU and GPU, the total number of instance that
 *    share ownership is equal to the number of instances that share ownership on the
 *    CPU both before you launch the kernel and after the kernel completes
 *  - There are 2 relevant observations to add to this discussion:
 *    - this discussion assumes that the primary constructor is always invoked to try
 *      to construct a \ref SharedHandle or \ref SharedDevPtr instance on the stack or
 *      the host's heap. Problems would arise if you tried to use placement new to
 *      initialize in device memory. For that matter, any attempt to track an instance
 *      of these constructs in one of the device's memory spaces that persists outside
 *      of a kernel would be problematic for the drawn conclusions (to my knowledge,
 *      isn't actually something we could really accomplish anyway without going out of
 *      our way to do something "bad" -- there's no practical benefit to try this).
 *    - It's also worth mentioning that even if we did want to track the total number of
 *      reference counts across the CPU and GPU, there isn't a straight-forward solution
 *      (if you pass a construct to a kernel by value, the copy constructor isn't
 *      technically invoked and the reference count won't get incremented). While there
 *      are workarounds, they aren't elegant.
 *
 *  What about our assumption? The degree of required care actually depends on the
 *  deleter callback. For example, deleters passed to \ref SharedDevPtr that are based
 *  upon `cudaFree` and `cudaFreeAsync` will have distinct requirements.
 *
 *  Why Use These Constructs
 *  ========================
 *  These constructs are most useful as building blocks in larger components. For
 *  example, aspects of Cholla's feedback and cooling modules make use of resource
 *  allocations for the entirety of a simulation run. These constructs make it easier
 *  to build up constructs in a composable manner.
 *
 *  While alternatives are possible, they typically involve either (i) implementing data
 *  structures using semantics like std::unique_ptr, or (ii) using global variables.
 *  - The first option causes problems with wrapping the full command (e.g. modelling
 *    cooling) in a std::function. We could work around this issue by creating an
 *    analogue of C++23's std::move_only_function or creating a custom command base
 *    class and passing around a pointer to that class. Even then, are more issues:
 *    - Unfortunately, move-semantics are a little intimidating for less-experienced
 *      C++ developers.
 *    - Furthermore, an object with unique_ptr-like semantics should not be directly
 *      be passed to a kernel by value because it would be inconsistent with standard
 *      move-semantics. While I'm not fundamentally opposed to implementing "an
 *      exception to the general rules of move-semantics," I think that would be a bad
 *      idea since I'm already concerned about some contributors learning about
 *      move-semantics. Thus, under this solution we should really be extracting all
 *      handles and pointers from a data structure before passing them to a kernel,
 *      which obviously hinders composability (which we desire for the cooling
 *      routines).
 *  - The second option isn't composable in a manner desired for the cooling routines.
 *    Every time we would want to add support for a different kind of cooling table, we
 *    would need to
 *    (everything is a special case and it makes testing more difficult)
 *
 *  In the future, the internals of \ref SharedDevPtr could be very useful. The View
 *  types adopted in libraries like Kokkos or Raja have the same shared object semantics
 *  as \ref SharedDevPtr. We could reuse the machinery to accomplish similar goals:
 *  - In the nearer term, we could rename \ref SharedDevPtr to something like
 *    ``Shared1DBuf`` use it to gradually replace every occurrence of
 *    \ref cuda_utilities::DeviceVector (obviously, we would need to add on more
 *    methods to ``Shared1DBuf``). This is beneficial since we could pass a
 *    ``Shared1DBuf`` directly to a kernel (making it possible to directly pass a
 *    \ref cuda_utilities::DeviceVector would be inconsistent with standard
 *    copy-semantics of a vector-like thing).
 *  - Longer term, if we wanted to support multi-dimensional views that internally
 *    convert a 3D index to a pointer-access (this could be very useful for reducing
 *    the amount of memory allocated when we use face-centered B-fields), we could
 *    start to make used of C++23's std::mdspan (backports of the library also exist).
 *    In that scenario, we would probably want to reuse our control-block logic to help
 *    implement a custom AccessorPolicy for std::mdspan that supports reference
 *    counting.
 */

namespace detail
{

// define logic for reference counting (on the host)
// - this default implementation uses built-in atomic operations provided with
//   the C++ standard library
// - this logic has been pull out of ControlBlock in case we want to take a
//   crack at implementing an alternative version in terms of OpenMP constructs
//   (this requires the use of OpenMP 3.1 or newer)
// - for context, the use of std::atomic with OpenMP **PROBABLY** works as you
//   would expect, but that is not strictly required by the OpenMP standard. In
//   practice compiler writers generally try to "do the right thing" and make
//   things work correctly. Furthermore, the kind of logic where this plays a
//   role is typically executed at the very start and end of Cholla's execution
//   (i.e. outside of all OpenMP blocks)
//
// optimization opportunity: the memory ordering is a little stricter than
//   necessary (out of an abundance of cautious), which introduces additional
//   overhead. In practice, this shouldn't matter very much in cholla since we
//   increment/decrement reference counts relatively infrequently; this happens
//   when initializing relevant datastructures and when they go out of scope
//   (which is almost entirely constrained to startup and shutdown)

using RefCountType = std::atomic<long>;
/*! atomically increment the reference count (return the old value) */
inline long ref_count_increment(RefCountType& count) noexcept { return count.fetch_add(1L, std::memory_order_seq_cst); }
/*! atomically decrement the reference count (return the old value) */
inline long ref_count_decrement(RefCountType& count) noexcept
{
  return count.fetch_add(-1L, std::memory_order_seq_cst);
}

/*! \brief Helps implement \ref SharedHandle & \ref SharedDevPtr
 *
 *  The term "control block," is used to describe the analogous data structure that is
 *  used in [typical implementations](https://en.cppreference.com/cpp/memory/shared_ptr#Implementation_notes)
 *  of `std::shared_ptr`
 *
 *  In more detail, a control block tracks:
 *  - the managed handle or pointer
 *  - the deleter
 *  - the reference count
 */
class ControlBlock
{
  RefCountType ref_count_;

 protected:
  virtual ~ControlBlock() noexcept {}

 public:
  ControlBlock() noexcept : ref_count_{1L} {}

  /*! \brief increment reference count */
  void increment_count() noexcept
  {
    long preincrement_val = ref_count_increment(ref_count_);
    CHOLLA_ASSERT(preincrement_val >= 1L, "invariant is violated");  // <- sanity check!
  }

  /*! \brief decrement ref count & trigger destructor of `this` if the count hits 0 */
  void decrement_count() noexcept
  {
    long predecrement_val = ref_count_decrement(ref_count_);
    CHOLLA_ASSERT(predecrement_val > 0L, "invariant is violated");  // <- sanity check!
    if (predecrement_val == 1L) delete this;
  }
};

template <typename HandleOrPtrType, typename Deleter>
class ControlBlockImpl : public ControlBlock
{
  HandleOrPtrType managed_;
  Deleter deleter_;

 public:
  ControlBlockImpl(HandleOrPtrType& m, Deleter& d) : ControlBlock(), managed_{m}, deleter_{d} {}

  ~ControlBlockImpl() { deleter_(managed_); }
};

}  // namespace detail

// down below, we move on to actually defining SharedHandle and SharedDevPtr.

// define the CALL_INCREMENT_COUNT and CALL_DECREMENT_COUNT macros
// -> these macros forward onto detail::ControlBlock::increment_count and
//    detail::ControlBlock::decrement_count when invoked on the host and do nothing
//    when invoked on the host
// -> this is the desired behavior we want (it's explained in more detail at the top of
//    this page)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  #define CALL_INCREMENT_COUNT(cb_ptr) /* DOES NOTHING ON DEVICE */
  #define CALL_DECREMENT_COUNT(cb_ptr) /* DOES NOTHING ON DEVICE */
#else
  #define CALL_INCREMENT_COUNT(cb_ptr) (cb_ptr)->increment_count()
  #define CALL_DECREMENT_COUNT(cb_ptr) (cb_ptr)->decrement_count()
#endif

/*! \brief Implements common methods of \ref SharedHandle and \ref SharedDevPtr
 *
 *  This macro exists due to the intrinsic similarity between \ref SharedHandle and
 *  \ref SharedDevPtr: a handle and a pointer are different ways to represent a
 *  to a resource. Thus, this macro provides definitions for methods that have the
 *  same structure in both class templates without repeating ourselves.
 *
 *  \param KLASS The name of the class template (i.e. SharedHandle or SharedDevPtr)
 */
#define DEFINE_COMMON_METHODS(KLASS)                                                                             \
                                                                                                                 \
  /* implement the copy constructor */                                                                           \
  template <typename T>                                                                                          \
  __host__ __device__ KLASS<T>::KLASS(const KLASS<T>& other) noexcept : wrapped_{other.wrapped_}, cb_{other.cb_} \
  {                                                                                                              \
    if (cb_ != nullptr) {                                                                                        \
      CALL_INCREMENT_COUNT(cb_);                                                                                 \
    }                                                                                                            \
  }                                                                                                              \
                                                                                                                 \
  /* implement the move constructor */                                                                           \
  template <typename T>                                                                                          \
  __host__ __device__ KLASS<T>::KLASS(KLASS<T>&& other) noexcept : wrapped_{other.wrapped_}, cb_{other.cb_}      \
  {                                                                                                              \
    other.set_empty_wrapped_();                                                                                  \
    other.cb_ = nullptr;                                                                                         \
  }                                                                                                              \
                                                                                                                 \
  /* implement the copy assignment operation */                                                                  \
  template <typename T>                                                                                          \
  __host__ __device__ KLASS<T>& KLASS<T>::operator=(const KLASS<T>& other) noexcept                              \
  {                                                                                                              \
    /* not only is this branch an optimization, it also prevents really bad bugs */                              \
    /* when `this` and `other` refer to the same instance and that instance is   */                              \
    /* is the ONLY owner of a resource... (if we decremented the reference count */                              \
    /* to zero before trying to increment it, that would be very bad)            */                              \
    if (cb_ != other.cb_) {                                                                                      \
      if (cb_ != nullptr) {                                                                                      \
        CALL_DECREMENT_COUNT(cb_);                                                                               \
      }                                                                                                          \
      cb_ = other.cb_;                                                                                           \
      if (cb_ != nullptr) {                                                                                      \
        CALL_INCREMENT_COUNT(cb_);                                                                               \
      }                                                                                                          \
    }                                                                                                            \
    wrapped_ = other.wrapped_;                                                                                   \
    return *this;                                                                                                \
  }                                                                                                              \
                                                                                                                 \
  /* implement the move assignment operation */                                                                  \
  template <typename T>                                                                                          \
  __host__ __device__ KLASS<T>& KLASS<T>::operator=(KLASS<T>&& other) noexcept                                   \
  {                                                                                                              \
    reset();     /*! <- release current ownership of resource (if any) */                                        \
    swap(other); /*! <- transfer any owned resource from other & make it empty */                                \
    return *this;                                                                                                \
  }                                                                                                              \
                                                                                                                 \
  template <typename T>                                                                                          \
  __host__ __device__ void KLASS<T>::swap(KLASS<T>& other) noexcept                                              \
  {                                                                                                              \
    decltype(wrapped_) tmp_wrapped = wrapped_;                                                                   \
    wrapped_                       = other.wrapped_;                                                             \
    other.wrapped_                 = tmp_wrapped;                                                                \
                                                                                                                 \
    detail::ControlBlock* tmp_cb = cb_;                                                                          \
    cb_                          = other.cb_;                                                                    \
    other.cb_                    = tmp_cb;                                                                       \
  }                                                                                                              \
                                                                                                                 \
  template <typename T>                                                                                          \
  __host__ __device__ void KLASS<T>::reset() noexcept                                                            \
  {                                                                                                              \
    /* reminder: if the reference count hits 0, the control block automatically: */                              \
    /*    - destroys wrapped_ (a copy of wrapped is tracked within the control   */                              \
    /*      block for this purpose)                                              */                              \
    /*    - calls delete on itself                                               */                              \
    if (cb_ != nullptr) {                                                                                        \
      CALL_DECREMENT_COUNT(cb_);                                                                                 \
      cb_ = nullptr;                                                                                             \
      set_empty_wrapped_();                                                                                      \
    }                                                                                                            \
  }

/*! \brief Wraps a handles while providing shared object semantics
 *  \ingroup sharedgrp
 *
 *  At a high level this analogous to shared_ptr, but for handles.
 *
 *  This type primarily exists in order to wrap the `cudaTextureObject_t` handle.
 */
template <typename HandleT>
class SharedHandle
{
  // perform some sanity checks:
  static_assert(std::is_arithmetic_v<HandleT> or std::is_aggregate_v<HandleT> or std::is_pointer_v<HandleT>);
  static_assert(not std::is_const_v<HandleT>);

  // data members:
  HandleT wrapped_;           ///< The wrapped handle
  detail::ControlBlock* cb_;  ///< The control block

  // helper method used by DEFINE_COMMON_METHODS to set the value of wrapped_ to the
  // appropriate value when a SharedHandle is "empty"
  // -> this method does nothing since the value of wrapped_ in an empty SharedHandle
  //    instance is explicitly undefined
  // -> this is defined for parity with SharedDevPtr
  __host__ __device__ __forceinline__ void set_empty_wrapped_() const noexcept {}

 public:
  // used for testing purposes
  typedef HandleT wrapped_ref_type;

  /*! \brief Default Constructor (constructs an empty instance)
   *
   *  \note
   *  A compelling case could be made for not having a default constructor, at all.
   *  For now, we define for the sake of symmetry with SharedPtr, and it makes
   *  composition of classes that use SharedHandle easier for less experienced C++
   *  developers
   */
  __host__ __device__ SharedHandle() : wrapped_{}, cb_{nullptr} {};

  /*! \brief Primary constructor
   *
   *  \p d is a callback function-like object that is used deallocate \p handle
   *  when all SharedHandle instances that wrap \p handle release ownership. The
   *  callback will be invoked as `d(handle)`
   *
   *  \param handle the handle to be managed by the constructed instance
   *  \param d A deleter instance used to destroy the handle
   *
   *  \warning
   *  Passing a \p handle that is already owned by another SharedHandle leads to
   *  undefined behavior
   */
  template <typename Deleter>
  __host__ SharedHandle(HandleT handle, Deleter d)
      : wrapped_{handle}, cb_{new detail::ControlBlockImpl<HandleT, Deleter>(handle, d)}
  {
  }

  /*! \brief Destructor */
  __host__ __device__ ~SharedHandle() noexcept { reset(); }

  // copy/move construction and assignment
  __host__ __device__ SharedHandle(const SharedHandle& other) noexcept;
  __host__ __device__ SharedHandle(SharedHandle&& other) noexcept;
  __host__ __device__ SharedHandle& operator=(const SharedHandle& other) noexcept;
  __host__ __device__ SharedHandle& operator=(SharedHandle&& other) noexcept;

  /*! \brief swap the contents of this and other */
  __host__ __device__ void swap(SharedHandle& o) noexcept;

  /*! \brief Release ownership of the owned resource (if any)
   *
   *  This is equivalent to `SharedHandle().swap(*this);`
   */
  __host__ __device__ void reset() noexcept;

  /*! \brief Return the stored handle */
  __host__ __device__ __forceinline__ HandleT get() const noexcept { return wrapped_; }

  /*! \brief returns ``false`` if ``this`` is empty */
  __host__ __device__ explicit operator bool() const noexcept { return cb_ != nullptr; }
};

// provide definitions for the remainder of methods
DEFINE_COMMON_METHODS(SharedHandle)

/*! \brief Wraps a device pointer while providing shared object semantics
 *  \ingroup sharedgrp
 *
 *  At a high level, this analogous to shared_ptr, but for device pointers.
 */
template <typename T>
class SharedDevPtr
{
  // data members:
  T* wrapped_;                ///< The wrapped pointer
  detail::ControlBlock* cb_;  ///< The control block

  // helper method used by DEFINE_COMMON_METHODS to set the value of wrapped_ to the
  // appropriate value when a SharedDevPtr is "empty"
  __host__ __device__ __forceinline__ void set_empty_wrapped_() noexcept { wrapped_ = nullptr; }

 public:
  // used for testing purposes
  typedef T* wrapped_ref_type;

  /*! \brief Default constructor (creates an "empty" instance) */
  __host__ __device__ SharedDevPtr() : wrapped_{nullptr}, cb_{nullptr} {}

  /*! \brief Primary constructor
   *
   *  \p d is a callback function-like object that is used deallocate \p ptr
   *  when all SharedDevPtr instances that wrap \p ptr release ownership. The
   *  callback will be invoked as `d(ptr)`
   *
   *  \param ptr the device pointer to be managed by the constructed instance
   *  \param d A deleter instance used to destroy the handle
   *
   *  \warning
   *  Passing a \p ptr that is already owned by another SharedDevPtr leads to undefined
   *  behavior
   */
  template <typename Deleter>
  __host__ SharedDevPtr(T* ptr, Deleter d) : wrapped_{ptr}, cb_{new detail::ControlBlockImpl<T*, Deleter>(ptr, d)}
  {
  }

  /*! \brief Destructor */
  __host__ __device__ ~SharedDevPtr() noexcept { reset(); }

  // copy/move construction and assignment
  __host__ __device__ SharedDevPtr(const SharedDevPtr& other) noexcept;
  __host__ __device__ SharedDevPtr(SharedDevPtr&& other) noexcept;
  __host__ __device__ SharedDevPtr& operator=(const SharedDevPtr& other) noexcept;
  __host__ __device__ SharedDevPtr& operator=(SharedDevPtr&& other) noexcept;

  /*! \brief swap the contents of this and other */
  __host__ __device__ void swap(SharedDevPtr& other) noexcept;

  /*! \brief Release ownership of the owned resource (if any)
   *
   *  This is equivalent to `SharedDevPtr().swap(*this);`
   */
  __host__ __device__ void reset() noexcept;

  /*! \brief Return the stored pointer */
  __host__ __device__ __forceinline__ T* get() const noexcept { return wrapped_; }

  /*! check if the wrapped pointer is a nullptr */
  __host__ __device__ explicit operator bool() const noexcept { return wrapped_ != nullptr; }

  /*! \brief Dereference the stored pointer
   *
   *  \note
   *  For less experienced C++ developers, this is the way to override the pointer
   *  dereference syntax. Suppose you have a SharedDevPtr instance called ``my_ptr``.
   *  This method lets you write ``*my_ptr``, which is equivalent to ``*(my_ptr.get())``
   */
  __device__ __forceinline__ T& operator*() const noexcept { return *wrapped_; }

  /*! \brief Dereference the stored pointer
   *
   *  \note
   *  For less experienced C++ developers, this is the way to override the pointer
   *  subscripting syntax. Suppose you have a SharedDevPtr instance called ``my_ptr``.
   *  This method lets you write ``my_ptr[i]`` (where ``i`` is a non-negative index),
   *  which is equivalent to ``(my_ptr.get())[i]``
   */
  __device__ __forceinline__ T& operator[](std::ptrdiff_t idx) const { return wrapped_[idx]; }
};

// provide definitions for the remainder of methods
DEFINE_COMMON_METHODS(SharedDevPtr)

// let's do some cleanup to avoid leaking of macros into other parts of the codebase
#undef CALL_INCREMENT_COUNT
#undef CALL_DECREMENT_COUNT
#undef DEFINE_COMMON_METHODS