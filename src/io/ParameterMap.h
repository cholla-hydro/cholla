#ifndef PARAMETERMAP_H
#define PARAMETERMAP_H

#include <climits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>

#include "../utils/error_handling.h"

// stuff inside this namespace is only meant to be used to implement ParameterMap
namespace param_details
{

/* defining a construct like this is a common workaround used to raise a compile-time error in the
 * else-branch of a constexpr-if statement. This is used to implement ``ParameterMap::try_get_``
 */
template <class>
inline constexpr bool dummy_false_v_ = false;

/* Kinds of errors from converting parameters to a type */
enum class TypeErr { none, generic, boolean, out_of_range };

/* function used to actually format/report the error message specified by the TypeErr enum */
[[noreturn]] void Report_TypeErr_(const std::string& param, const std::string& str, const std::string& dtype,
                                  TypeErr type_convert_err);

/* @{
 * helper functions that try to interpret a string as a given type.
 *
 * This returns the associated value if it has the specified type. If ``type_mismatch_is_err`` is
 * true, then the program aborts with an error if the string is the wrong type. When
 * ``type_mismatch_is_err``, this simply returns an empty result.
 */
param_details::TypeErr try_int64_(const std::string& str, std::int64_t& val);
param_details::TypeErr try_double_(const std::string& str, double& val);
param_details::TypeErr try_bool_(const std::string& str, bool& val);
param_details::TypeErr try_string_(const std::string& str, std::string& val);

// special case to make people's lives easier
inline param_details::TypeErr try_int_(const std::string& str, int& val)
{
  std::int64_t tmp;
  TypeErr err = try_int64_(str, tmp);
  if ((err == param_details::TypeErr::none) and (INT_MIN <= tmp) && (tmp <= INT_MAX)) {
    val = int(tmp);
    return TypeErr::none;
  }
  return (err == TypeErr::none) ? TypeErr::out_of_range : err;
}
/* @} */
}  // namespace param_details

/*!
 * \brief A class that provides map-like access to parameter files.
 *
 * After construction, the collection of parameters and associated values can not be mutated.
 * However, the class is not entirely immutable; internally it tracks whether parameters have been
 * accessed.
 *
 * In contrast to formats like TOML, JSON, & YAML, the parameter files don't have syntactic typing
 * (i.e. where the syntax determines formatting). In this sense, the format is more like ini files.
 * As a consequence, we internally store the parameters as strings. The access API explicitly
 * converts them to the user-specified type.
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
    std::string param_str;
    bool accessed;
  };

 private:  // attributes
  std::map<std::string, ParamEntry> entries_;

 public:  // interface methods
  /* Reads parameters from a parameter file and arguments.
   *
   * \note
   * We pass in a ``std::FILE`` object rather than a filename-string because that makes testing
   * easier.
   */
  ParameterMap(std::FILE* fp, int argc, char** argv, bool close_fp = false);

  /* An overload for the constructor */
  ParameterMap(const std::string& fname, int argc, char** argv);

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

  /* This is a temporary function to help ease the transition to the new parsing approach. */
  template <typename LegacyParseParamFn>
  void pass_entries_to_legacy_parse_param(LegacyParseParamFn& f)
  {
    for (auto& kv_pair : entries_) {
      const char* name  = kv_pair.first.c_str();
      const char* value = (kv_pair.second).param_str.c_str();

      // pass the parameter name and (unparsed) value to the legacy function. Record if used.
      bool rslt = f(name, value);
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

  const std::string& str = (keyvalue_pair->second).param_str;  // string associate with param

  // convert the string to the specified type and store it in out
  T val{};                       // default constructed
  param_details::TypeErr err{};  // reports errors
  const char* dtype_name;        // used for formatting errors (we use a const char* rather than a
                                 // std::string so we can hold string-literals)

  // The branch of the following if-statement is picked at compile-time
  if constexpr (std::is_same_v<T, bool>) {
    err        = param_details::try_bool_(str, val);
    dtype_name = "bool";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    err        = param_details::try_int64_(str, val);
    dtype_name = "int64_t";
  } else if constexpr (std::is_same_v<T, double>) {
    err        = param_details::try_double_(str, val);
    dtype_name = "double";
  } else if constexpr (std::is_same_v<T, std::string>) {
    err        = param_details::try_string_(str, val);
    dtype_name = "string";
  } else if constexpr (std::is_same_v<T, int>) {
    err        = param_details::try_int_(str, val);
    dtype_name = "int";
  } else {
    static_assert(param_details::dummy_false_v_<T>,
                  "template type can only be bool, int, std::int64_t, double, or std::string.");
  }

  // now do err-handling/value return
  if (err != param_details::TypeErr::none) {
    if (is_type_check) return {};  // return empty option
    param_details::Report_TypeErr_(param, str, dtype_name, err);
  }

  if (not is_type_check) (keyvalue_pair->second).accessed = true;  // record parameter-access
  return {val};
}

#endif /* PARAMETERMAP_H */