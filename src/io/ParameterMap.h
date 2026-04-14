#ifndef PARAMETERMAP_H
#define PARAMETERMAP_H

#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <variant>

#include "../utils/error_handling.h"

// stuff inside this namespace is only meant to be used to implement ParameterMap
namespace param_details
{

/*! Kinds of errors from parsing */
enum class ParseErr { none, generic, out_of_range };

/*! function used to actually format/report the error message specified by the ParseErr enum */
[[noreturn]] void Report_ParseErr_(const std::string& param, const std::string& str, const std::string& dtype,
                                   ParseErr parse_err);

/*! Represents the parameter value. */
struct Value {
  /*! This is a type-safe union holding the value.
   *
   *  \note
   *  We wrap the Variant inside a class for 2 reasons: (i) to try to make an eventual
   *  transition to toml++ easier and (ii) because working directly directly with
   *  std::variant can be tricky for less experienced C++ devs
   */
  using Variant = std::variant<std::string, bool, int64_t, double>;

 private:
  Variant v_;

 public:
  Value() = default;
  explicit Value(Variant v) : v_(v) {}

  /// \name Type checks
  ///@{
  bool is_string() const noexcept { return std::holds_alternative<std::string>(v_); }
  bool is_integer() const noexcept { return std::holds_alternative<int64_t>(v_); }
  bool is_floating_point() const noexcept { return std::holds_alternative<double>(v_); }
  bool is_boolean() const noexcept { return std::holds_alternative<bool>(v_); }
  ///@}

  /*! Return the toml representation of the value
   *
   *  This primarily exists to help implement \ref ParameterMap::pass_entries_to_legacy_parse_param
   *  (which will be deleted in PR #495). We also use it for writing error messages/warnings.
   */
  std::string toml_repr() const;

  template <typename T>
  std::optional<T> value_exact() const
  {
    const T* tmp = std::get_if<T>(&v_);
    if (tmp == nullptr) return std::nullopt;
    return {*tmp};
  }

  const char* type_name() const noexcept
  {
    return std::visit(
        [](auto&& arg) -> const char* {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            return "integer";
          } else if constexpr (std::is_same_v<T, double>) {
            return "floating point";
          } else if constexpr (std::is_same_v<T, std::string>) {
            return "string";
          } else if constexpr (std::is_same_v<T, bool>) {
            return "boolean";
          } else {
            static_assert(always_false<T>, "unexpected type.");
          }
          return nullptr;  // <- this should not be necessary...
        },
        v_);
  }
};

template <typename T>
constexpr bool support_lossless_conversion_(int64_t v)
{
  int64_t lower, upper;
  if constexpr (std::is_floating_point_v<T>) {
    int64_t n_mantissa_digits_plus_one = int64_t{std::numeric_limits<T>::digits};
    // calclate 2^n_mantissa_digits_plus_one (https://stackoverflow.com/a/3793950)
    upper = int64_t{2} << n_mantissa_digits_plus_one;
    lower = -1 * upper;
  } else if constexpr (std::is_integral_v<T> and std::is_signed_v<T> and (sizeof(int64_t) >= sizeof(T))) {
    upper = int64_t{std::numeric_limits<T>::max()};
    lower = int64_t{std::numeric_limits<T>::min()};
  } else {
    static_assert(always_false<T>, "template has unexpected type.");
  }
  return (lower <= v) and (v <= upper);
}
}  // namespace param_details

/*!
 * \brief A class that provides map-like access to parameter files.
 *
 * After construction, the collection of parameters and associated values can not be mutated.
 * However, the class is not entirely immutable; internally it tracks whether parameters have been
 * accessed.
 *
 * \note
 * We primarily support 4 datatypes: ``bool``, ``std::int64_t``, ``double``, ``std::string``.
 * - For convenience, we provide support for internally casting values to ``int``.
 * - We currently do not provide support for internally casting values to ``float``.
 * The reason for this distinction is that within the overlapping interval of values represented by
 * both``int`` and ``std::int64_t``, values are represented with equal levels of accuracy. In
 * contrast, for the overlapping interval of values represented by both ``float`` and ``double``,
 * the latter represents some values with greater accuracy.
 */
class ParameterMap
{
 public:
  struct ParamEntry {
    param_details::Value val;
    bool accessed;
  };

 private:  // attributes
  std::map<std::string, ParamEntry> entries_;

 public:  // interface methods
  /*! Reads parameters from a parameter file and arguments.
   *
   *  \param fp The file that is read from
   *  \param n_param_override_args The number of elements in \p param_override_args
   *  \param param_override_args An array of parameter-override command line arguments
   *  \param close_fp Indicates whether to close \p fp on completion
   *
   *  \note
   *  We pass in a ``std::FILE`` object rather than a filename-string because that makes testing
   *  easier.
   */
  ParameterMap(std::FILE* fp, int n_param_override_args, char** param_override_args, bool close_fp = false);

  /*! An overload for the primary constructor */
  ParameterMap(const std::string& fname, int n_param_override_args, char** param_override_args);

  /* queries the number of parameters (mostly for testing purposes) */
  std::size_t size() { return entries_.size(); }

  /* queries whether the parameter exists. */
  bool has_param(const std::string& param) { return entries_.find(param) != entries_.end(); }

  /* queries whether the parameter exists and if it has the specified type.
   *
   * \note
   * The result is always the same as ``has_param``, when ``T`` is ``std::string``.
   */
  template <typename T>
  bool param_has_type(const std::string& param)
  {
    return try_get_<T>(param, true).has_value();
  }

  /* Retrieves the value associated with the specified parameter. If the
   * parameter does not exist or does not have the specified type, then the
   * program aborts with an error.
   *
   * \tparam The expected type of the parameter-value
   *
   * \note The name follows conventions of std::optional
   */
  template <typename T>
  T value(const std::string& param)
  {
    std::optional<T> result = try_get_<T>(param, false);
    if (not result.has_value()) {
      CHOLLA_ERROR("The \"%s\" parameter was not specified.", param.c_str());
    }
    return result.value();
  }

  /* @{
   * If the specified parameter exists, retrieve the associated value, otherwise return default_val.
   * If the associated value does not have the specified type, the program aborts with an error.
   *
   * \param param The name of the parameter being queried.
   * \param default_val The value to return in case the parameter was not defined.
   *
   * \note
   * This is named after std::optional::value_or. It's my intention to replace this with a single
   * template, but this is good enough for now!
   *
   * \note
   * Except when considering strings, the return type is always the same as the default value
   */
  bool value_or(const std::string& param, bool default_val)
  {
    return try_get_<bool>(param, false).value_or(default_val);
  }

  int value_or(const std::string& param, int default_val) { return try_get_<int>(param, false).value_or(default_val); }

  std::int64_t value_or(const std::string& param, std::int64_t default_val)
  {
    return try_get_<std::int64_t>(param, false).value_or(default_val);
  }

  double value_or(const std::string& param, double default_val)
  {
    return try_get_<double>(param, false).value_or(default_val);
  }

  std::string value_or(const std::string& param, const std::string& default_val)
  {
    return try_get_<std::string>(param, false).value_or(default_val);
  }

  std::string value_or(const std::string& param, const char* default_val)
  {
    return try_get_<std::string>(param, false).value_or(default_val);
  }
  /* @} */

  /* Warns about parameters that have not been accessed with the ``value`` OR ``value_or`` methods.
   *
   * \param ignore_params a set of parameter names that should never be reported as unused
   * \param abort_on_warning when true, the warning is reported as error that causes the program to
   *    abort. Default is false.
   * \param suppress_warning_msg when true, the warning isn't actually printed (this only exists for
   *    testing purposes)
   * \returns the number of unused parameters
   */
  int warn_unused_parameters(const std::set<std::string>& ignore_params, bool abort_on_warning = false,
                             bool suppress_warning_msg = false) const;

  /*! This is a temporary function to help ease the transition to the new parsing approach.
   *
   *  \note This will be deleted in PR #495
   */
  template <typename LegacyParseParamFn>
  void pass_entries_to_legacy_parse_param(LegacyParseParamFn& f)
  {
    for (auto& kv_pair : entries_) {
      const std::string& name = kv_pair.first;
      std::string value       = (kv_pair.second).val.toml_repr();

      // pass the parameter name and (unparsed) value to the legacy function. Record if used.
      bool rslt = f(name.c_str(), value.c_str());
      if (rslt) (kv_pair.second).accessed = true;
    }
  }

  /*! Aborts with an error message if one or more of the parameters in the specified table has been used or has not
   *  been used. The precise details depend on the `expect_unused` argument.
   *
   *  \note
   *  It may be better if this were a function that operated on ParameterMap rather than a method */
  void Enforce_Table_Content_Uniform_Access_Status(std::string table_name, bool expect_unused) const;

 private:  // private helper methods
  /* helper function template that tries to retrieve values associated with a given parameter.
   *
   * This returns the associated value if it exists and has the specified type. The returned
   * value is empty if the parameter doesn't exist. If the It can also be empty when type_abort is
   * ``true`` and the specified type doesn't match the parameter (and is a type a parameter can
   * have).
   */
  template <typename T>
  std::optional<T> try_get_(const std::string& param, bool is_type_check);
};

template <typename T>
std::optional<T> ParameterMap::try_get_(const std::string& param, bool is_type_check)
{
  auto keyvalue_pair = entries_.find(param);
  if (keyvalue_pair == entries_.end()) return {};  // return emtpy option

  const param_details::Value& param_val = (keyvalue_pair->second).val;

  // try to extract the underlying value
  std::optional<T> out{};  // default constructed
  bool out_of_range = false;
  const char* dtype_name;  // used for formatting errors (we use a const char* rather than a
                           // std::string so we can hold string-literals)

  // The branch of the following if-statement is picked at compile-time
  if constexpr (std::is_same_v<T, bool>) {
    out        = param_val.value_exact<bool>();
    dtype_name = "bool";
  } else if constexpr (std::is_same_v<T, std::string>) {
    out        = param_val.value_exact<std::string>();
    dtype_name = "string";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    out        = param_val.value_exact<int64_t>();
    dtype_name = "int64_t";
  } else if constexpr (std::is_same_v<T, double>) {
    out = param_val.value_exact<double>();
    if (not out.has_value()) {
      std::optional<int64_t> tmp = param_val.value_exact<int64_t>();
      if (tmp.has_value()) {
        if (param_details::support_lossless_conversion_<T>(*tmp)) {
          out = static_cast<double>(*tmp);
        } else {
          out_of_range = true;
        }
      }
    }
    dtype_name = "double";
  } else if constexpr (std::is_same_v<T, int>) {
    std::optional<int64_t> tmp = param_val.value_exact<int64_t>();
    if (tmp.has_value()) {
      if (param_details::support_lossless_conversion_<T>(*tmp)) {
        out = {static_cast<int>(*tmp)};
      } else {
        out_of_range = true;
      }
    }
    dtype_name = "int";
  } else {
    static_assert(always_false<T>, "template type can only be bool, int, std::int64_t, double, or std::string.");
  }

  // now do err-handling/value return
  if (not out.has_value()) {
    if (is_type_check) return std::nullopt;  // return empty option
    const char* val_type = param_val.type_name();
    const char* reason   = out_of_range ? "out of range" : "invalid conversion";
    CHOLLA_ERROR("error interpretting the %s value associated with the \"%s\" as a %s value: %s\n", val_type,
                 param.c_str(), dtype_name, reason);
  }

  if (not is_type_check) (keyvalue_pair->second).accessed = true;  // record parameter-access
  return out;
}

#endif /* PARAMETERMAP_H */