/*! \file
 *  Defines the ModelCollection type
 */

#pragma once

#include <type_traits>
#include <variant>
#include <vector>

#include "../io/ParameterMap.h"
#include "../utils/error_handling.h"  // always_false
#include "disk_galaxy.h"

/*! \defgroup modelgrp Model Group
 *
 *  This group describes the basic infrastructure pertaining to defining and managing
 *  "models."
 *
 *  What is a Model?
 *  ================
 *  At a high-level a "model" is an entity, tracked by \ref ModelCollection, that
 *  aggregates information. A "model" is always configured by a dedicated table of
 *  parameters from the parameter file. It **may** provide some functionality based on
 *  these parameters.
 *
 *  \important
 *  Models have 3 important features:
 *  1. models should not have any kind of mutable state (more on this below)
 *  2. relatedly, models should generally not encode information related to the size of
 *     a single process's grid. In other words, a model should generally look and act
 *     the same for every grid run as part of the simulation in a hypothetical future
 *     where Cholla uses AMR (we'll also touch on this below).
 *  3. models are initialized before the vast majority of other simulation components
 *
 *  While a "model" could be used for various purposes, the most compelling using-case
 *  is for encoding information that must be accessed from multiple parts of the
 *  codebase. We go into more detail about the motivation below and subsequently, we
 *  describe some example uses.
 *
 *  \note
 *  To transition from compilation macros towards runtime parameters, it **may** be
 *  useful to introduce a "model" that is initially configured based on macros, replace
 *  all code using compilation macros to make decisions based on the information
 *  embedded in the compilation macro. transition towards configuring it with runtime
 *  macros.
 *
 *  Motivation for models
 *  =====================
 *  As noted above, the primary motivation for creating a model is when you have a set
 *  of parameters/information that must be accessed in 2 or more distinct components of
 *  Cholla.
 *
 *  There are 2 other ways of doing this: (i) simply access the parameter value from the
 *  \ref ParameterMap in every component where the parameter is used OR (ii) using
 *  global objects to track the logic in a "model" object. We review why these
 *  alternatives are less desirable below.
 *
 *  Why not access a single parameter value from various simulation components?
 *  ---------------------------------------------------------------------------
 *  Problems often arise from reading parameter values in more than one location because
 *  simulation codes evolve over time and often introduce more-sophisticated logic. When
 *  this happens, the interpretation of existing parameters may change and it may be
 *  hard to ensure that the interpretation is appropriate in every location where we
 *  read the value.
 *
 *  Let's consider 2 scenarios:
 *  1. It's common for a parameter to just expect positive values.
 *     - At face value, it may seem fine to read this in multiple places.
 *     - Later on, someone may decides to assign special significance to a value of 0.
 *     - more concretely imagine that someone is modelling the potential of a disk and
 *       a parameter ``zh`` encodes the scale height. Later on someone else could come
 *       along and decide that the ``zh=0`` means that we use the potential of an
 *       infinitely thin disk.
 *  2. It's also common to introduce a new parameter that changes how an existing
 *     parameter is interpretted.
 *
 *  Why prefer models to global objects
 *  -----------------------------------
 *  What we now track as models were historically tracked as global objects. However,
 *  there are 3 advantages to encoding this information in a model that is tracked by
 *  a \ref ModelCollection:
 *
 *  1. The type information is commonly associated with modules of logic that can be
 *     entrirely disabled. Thus, you need to decide what Cholla does when you don't
 *     want to enable that logic.
 *     - The easy thing to do with global variables is to use conditional compilation
 *       to only define the global variable when a certain pre-compiler macro is present
 *     - When encoding the logic inside a model, you simply need to decide whether or
 *       not to insert the model into a \ref ModelCollection
 *  2. When using a global variable, there is some ambiguity about when exactly to
 *     initialize the global variable. Different modules will probably make different
 *     choices (which could be problematic if you try to access the encoded information
 *     across different simulation components). In contrast, all model objects are
 *     initialized at a certain time during startup.
 *
 *  Examples
 *  ========
 *  2 sample use-cases that Cholla has (or will have) include:
 *  1. a galaxy model:
 *     - it encodes information needed to initialize gas and, in some cases, stars
 *     - in certain configurations, it encodes information needed to invoke the
 *       ParisGalactic gravity and the corresponding boundary conditions
 *     - in certain configurations, it directly sets the static potential. Ideally, this
 *       will directly set source terms related to the static potential in all cholla
 *       configurations in the future.
 *  2. a cloud-wind model:
 *     - it encodes information needed to initialize gas
 *     - the encoded initial cloud density will needs to be accessed by a method that
 *       implements frame-tracking.
 *     - the encoded wind velocity could be used for setting boundary conditions
 *
 *  It may be instructive to consider subclasses of the `Physics` type in Enzo-E that
 *  serve a similar purpose:
 *  - a fluid_props object that tracks assorted fluid properties
 *    - properties include:
 *      - EOS related quantities. This is namely adiabatic index, but the premise was
 *        to also support other equation of states (e.g. the isothermal EOS)
 *      - dual-energy related info (e.g. whether its enabled and the eta values)
 *      - the mean molecular weight (used without chemistry)
 *      - floors that may get applied in various sections of the codebase (e.g. density
 *        floor or pressure floor).
 *    - if we did something similar, it might make sense to divide such logic.
 *  - a gravity object that simply encodes an arbitrary Gravitational constant for
 *    non-cosmological simulations (useful in test problems).
 *  - a cosmology object that tracks the cosmological parameters (the ship has
 *    probably sailed for us to do something like that in Cholla)
 *
 *  Lower level description of a model
 *  ==================================
 *  At this time, a model is an arbitrary type that has a constructor that accepts a
 *  reference to \ref ParameterMap (in the future, we *may* want to consider also
 *  providing access to already initialized models).
 *
 *  Currently, \ref ModelCollection tracks models wrapped by ``std::variant`` (this
 *  is essentially a safe union) instances, because that was simple for a small number
 *  of models.
 *
 *  In the future, we **PROBABLY** want to switch to a solution where
 *  \ref ModelCollection tracks models as pointers to a common base class.
 *  - this alternative is better as the number of models grow from an incremental
 *    recompilation perspective
 *  - it is also beneficial if we standardize things such that we use commonly provide
 *    callbacks for triggering kernels that set model-specific boundary conditions,
 *    that perform model-specific components of the gravity solver, or that implement
 *    model-specific source terms. (This may or may not make sense)
 *  - this strategy would involve something like RTTI (Run-Time Type Information). For
 *    performance reasons, we would probably want to do something like
 *    https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html
 *    rather than use C++'s built-in RTTI.
 *  - We would need to decide whether the pointer should point to the model itself or a
 *    class template that wraps the model (it depends on whether we want to be able to
 *    directly access the model itself on GPUs).
 *
 *  Models should be immutable and identical across processes
 *  =========================================================
 *  As we noted above, "models" should satisfy the following invariants:
 *  1. a model is immutable (calling any associated methods won't change the state)
 *  2. a model should be identical for all grids in the simulation. You should write
 *     things in a hypothetical future where Cholla supports AMR.
 *
 *  If there is mutable state associated within your model, you should be tracking it
 *  elsewhere. At the moment, the best place to track this information is probably
 *  within the \ref Header instance tracked by a \ref Grid instance (and obviously you
 *  need to update the logic for serializing and deserializing that data). Ideally, we
 *  will implement a better system for this logic in the future.
 *
 *  \warning
 *  At the time, your code may work even if you violate these invariants. But, they
 *  will definitely break in the future if we do any kind of refactoring. Unless the fix
 *  is obvious and not invasive, we may choose to simply comment-out/delete any models
 *  that violate this invariant.
 *
 *  The primary impetus for these invariants is (de)serialization (at restarts and when
 *  migrating grids in hypothetical AMR sims to load balance). If possible want to avoid
 *  writing this logic for \ref ModelCollection since triggering the deserialization
 *  logic that creates the appropriate type of model can be a little messy/verbose, the
 *  data structures initialized within a model from parameters can be quite
 *  sophisticated (see \ref ClusteredDiskGalaxy), and most mutable state is typically
 *  quite simple. This is all quite doable, but it's simpler to avoid implementing it
 *  by only constructing models once (at startup) from the parameter file and handling
 *  the mutable state separately.
 */
///@{

namespace model_detail
{

// this is just a placeholder model type until we have 2 or more models
struct DummyModel {
  explicit DummyModel(ParameterMap& pmap) {}
};

// a type-safe union that can represent all model types
// -> to add a new kind of model, append it to the list of template arguments
// -> todo: consolidate DiskGalaxy and ClusteredDiskGalaxy into a single class
using model_variant = std::variant<DummyModel, ClusteredDiskGalaxy>;

// define logic to check if a type T is an allowed type of a std::variant
template <typename T, typename variant>
struct isVariantAlternativeType;

template <typename T, typename... Types>
struct isVariantAlternativeType<T, std::variant<Types...>> : public std::disjunction<std::is_same<T, Types>...> {
};

// evaluates to whether a type T is a known variant type
template <class T>
constexpr bool is_model_type = isVariantAlternativeType<T, model_variant>::value;

}  // namespace model_detail

/*! \brief Holds a collection of models */
class ModelCollection
{
  /// vector of models
  std::vector<model_detail::model_variant> vec_;

 public:
  /*! \brief default constructor */
  ModelCollection() {}

  /*! \brief primary constructor */
  explicit ModelCollection(ParameterMap& pmap) {}

  ModelCollection(ModelCollection&&)            = default;
  ModelCollection& operator=(ModelCollection&&) = default;

  // we forbid copy construction and copy assignment to retain flexibility (in case
  // we convert vec_ to hold std::unique_ptr instances)
  ModelCollection(const ModelCollection&)            = delete;
  ModelCollection& operator=(const ModelCollection&) = delete;

  /*! \brief Try to retrieve a model of the specified type
   *
   *  \returns A pointer to the specified model-type (if applicable).
   */
  template <typename T>
  const T* try_get() const
  {
    // here is a dummy implementation to be used while converting all existing logic to
    // try to access clustered disk galaxy through this approach
    if constexpr (std::is_same_v<T, ClusteredDiskGalaxy>) {
      return &galaxies::get_MW_model();
    } else {
      static_assert(always_false<T>, "unknown model type");
    }

    // here is the real implementation
    // ===============================
    // static_assert(model_detail::is_model_type<T>, "T isn't a known model type");
    // for (const model_detail::model_variant& v : vec_) {
    //  const T* ptr = std::get_if<T>(v);
    //  if (ptr != nullptr) return ptr;
    //}
    // return nullptr;
  }
};
///@}  // <- close the doxygen modelgrp