#include "../io/ParameterMap.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "../global/global.h"      // MAXLEN
#include "../io/io.h"              // chprintf, to_string_exact
#include "../io/to_from_string.h"  // io::try_parse_param_str, io::encode_toml_str
#include "../utils/error_handling.h"

namespace param_details
{

std::string Value::toml_repr() const
{
  return std::visit(
      [](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
          return std::to_string(static_cast<long long>(arg));
        } else if constexpr (std::is_same_v<T, double>) {
          return to_string_exact(arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return io::encode_toml_str(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
          return arg ? "true" : "false";
        } else {
          static_assert(always_false<T>, "unexpected type.");
        }
        return {};  // <- this should not be necessary...
      },
      v_);
}

static std::optional<Value> try_parse_number_(std::string str, bool& range_err)
{
  // first, try to parse str as a string
  char* ptr_end    = nullptr;
  errno            = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  long long tmp_ll = std::strtoll(str.data(), &ptr_end, 10);  // the last arg specifies base-10

  // before doing anything else, cache the value of errno (so we don't accidentally overwrite it)
  // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
  bool ERANGE_ll = (errno == ERANGE);

  // check whether the full string was parsed. For context:
  // - when str.data() == ptr_end, then no conversion was performed.
  // - when (str.data() + str.size()) != ptr_end, str could hold a float or look like "123abc"
  if ((str.data() + str.size()) == ptr_end) {
    if (ERANGE_ll) {
      range_err = true;
      return {};
#if (LLONG_MIN != INT64_MIN) || (LLONG_MAX != INT64_MAX)
    } else if ((tmp_ll < INT64_MIN) and (tmp_ll > INT64_MAX)) {
      range_err = true;
      return {};
#endif
    } else {
      range_err = false;
      Value::Variant variant{std::int64_t(tmp_ll)};
      return {Value(variant)};
    }
  }

  // next, lets try to parse as a double (it's important we do this second!)
  ptr_end       = nullptr;
  errno         = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  double tmp_fp = std::strtod(str.data(), &ptr_end);

  // before doing anything else, cache the value of errno (so we don't accidentally overwrite it)
  // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
  bool ERANGE_fp = (errno == ERANGE);

  // check whether the full string was parsed. For context:
  // - when str.data() == ptr_end, then no conversion was performed.
  // - when (str.data() + str.size()) != ptr_end, str could hold a float or look like "123abc"
  if ((str.data() + str.size()) == ptr_end) {
    if (ERANGE_fp) {
      range_err = true;
      return {};
    } else {
      range_err = false;
      Value::Variant variant{tmp_fp};
      return {Value(variant)};
    }
  }

  // the string doesn't hold a number
  range_err = false;
  return {};
}

static std::optional<Value> try_parse_val_(std::string buf, bool& range_err)
{
  std::size_t buf_size = buf.size();

  range_err = false;
  if (buf_size == 0) return {};

  // try to parse as a boolean
  if (buf == "true") {
    Value::Variant tmp{true};
    return {Value(tmp)};
  } else if (buf == "false") {
    Value::Variant tmp{false};
    return {Value(tmp)};
  }

  // try to parse buf as a string
  std::pair<std::size_t, std::string> str_rslt = io::try_parse_param_str(buf);
  if (str_rslt.first == buf.size()) {
    // the if statement avoids the scenario where buf looks like
    // > 'hi'there
    // and rslt.second holds "hi"
    Value::Variant tmp{str_rslt.second};
    return {Value(tmp)};
  }

  // finally, try to parse buf as a number
  return try_parse_number_(buf, range_err);
}

}  // namespace param_details

namespace
{  // stuff inside an anonymous namespace is local to this file

void rstrip(std::string_view& s)
{
  std::size_t cur_len = s.size();
  while ((cur_len > 0) and std::isspace(s[cur_len - 1])) {
    cur_len--;
  }
  if (cur_len < s.size()) s = s.substr(0, cur_len);
}

/*! \brief Modifies the string_view to remove trailing and leading whitespace.
 *
 *  \note
 *  Since this is a string_view, we don't actually mutate any characters
 */
void my_trim(std::string_view& s)
{
  /* Trim left side */
  std::size_t start           = 0;
  const std::size_t max_start = s.size();
  while ((start < max_start) and std::isspace(s[start])) {
    start++;
  }
  if (start > 0) s = s.substr(start);

  rstrip(s);
}

/*! Aggregates information about the key-value pair parsed by \ref Try_Extract_Key_Value.
 *
 *  When there is an error, \ref maybe_value is empty, \ref n_key_segments is `0`, and
 *  \ref error_msg holds an error message.
 */
struct KeyValueExtraction {
  std::string key;
  std::optional<param_details::Value> maybe_value;
  int n_key_segments;
  std::string error_msg;
};

/*! \brief Try to extract the parts of buffer that refers to a parameter name and the
 *  associated parameter value.
 */
KeyValueExtraction Try_Extract_Key_Value(std::string_view full_view)
{
  std::size_t pos                 = 0;
  io::KeyParseRslt key_parse_rslt = io::try_parse_param_key(full_view, pos);
  if (key_parse_rslt.n_segments == 0) {
    return {"", {}, 0, key_parse_rslt.val};
  }
  pos = key_parse_rslt.pos;

  // explicitly mimic the old behavior (forbid a space between the key & '=' character)
  if (full_view.size() == pos or full_view[pos] != '=') {
    return {"", {}, 0, "key should be immediately followed by a '=' character"};
  } else if (pos + 1 == full_view.size()) {
    return {"", {}, 0, "'=' is the last character on a line"};
  }
  pos++;
  std::string_view value_view = full_view.substr(pos);
  my_trim(value_view);
  std::string value_string(value_view);

  // let's try to explicitly parse the parameter
  bool range_err                                  = false;
  std::optional<param_details::Value> maybe_value = param_details::try_parse_val_(value_string, range_err);

  std::string error_msg;
  if (not maybe_value.has_value()) {
    error_msg = "error interpretting the value associated with the " + key_parse_rslt.val + " parameter";
    if (range_err) error_msg += ": value is out of range";
  }
  return {key_parse_rslt.val, maybe_value, key_parse_rslt.n_segments, error_msg};
}

/*! \brief Object used to read in lines from a `FILE*`
 *
 *  This primarily exists to help with error/warning formatting
 */
struct FileLineStream {
  long long line_num;
  FILE* fp;
  char *s, buff[256];

  FileLineStream(FILE* fp) : line_num{-1}, fp{fp}, s{nullptr} {}

  bool next()
  {
    this->line_num++;
    return (this->s = fgets(this->buff, sizeof this->buff, this->fp)) != NULL;
  }

  // formats a slightly more generic error
  [[noreturn]] void error(std::string reason) const
  {
    std::string msg = "parsing problem\n";
    // specify the location
    msg += this->location_description_();
    // include the reason
    msg += "   err: ";
    msg += reason;
    throw std::runtime_error(msg);
  }

  // more-detailed error formatting
  [[noreturn]] void error(std::string reason, std::string full_name, bool is_table_header) const
  {
    std::string msg = (is_table_header) ? "table-header parsing problem\n" : "parameter-name parsing problem\n";
    // give table/parameter name
    msg += (is_table_header) ? "   table-name: " : "   full-parameter-name: ";
    msg += full_name;
    msg += '\n';
    // specify the location
    msg += this->location_description_();
    // include the reason
    msg += "   err: ";
    msg += reason;
    throw std::runtime_error(msg);
  }

  void warn(std::string reason) const
  {
    std::string msg                  = "parameter-file parsing warning\n";
    std::string location_description = this->location_description_();
    chprintf("parameter-file parsing warning\n%s   message:%s\n", location_description.c_str(), reason.c_str());
  }

 private:
  std::string location_description_() const
  {
    std::string out = "   on line ";
    out += std::to_string(this->line_num);
    out += "of the parameter file: ";
    out += this->s;
    out += '\n';
    return out;
  }
};

struct CliLineStream {
  char** next_arg;
  char** stop;
  char* s;

  CliLineStream(int n_param_override_args, char** param_override_args)
      : next_arg{param_override_args}, stop{param_override_args + n_param_override_args}, s{nullptr} {};

  bool next()
  {
    if (this->next_arg == this->stop) return false;
    this->s = *this->next_arg;
    this->next_arg++;
    return true;
  }

  // formats a slightly more generic error
  [[noreturn]] void error(std::string reason) const
  {
    std::string msg = "parameter assignment parsing problem\n";
    // specify the location
    msg += "   from cli arg: ";
    msg += s;
    // include the reason
    msg += "   err: ";
    msg += reason;
    throw std::runtime_error(msg);
  }

  [[noreturn]] void error(std::string reason, std::string full_name, bool is_table_header) const
  {
    CHOLLA_ASSERT(not is_table_header, "something went very wrong - can't parse table header from cli");

    std::string msg = "parameter-name parsing problem\n";
    // give table/parameter name
    msg += "   full-parameter-name: ";
    msg += full_name;
    msg += '\n';
    // specify the location
    msg += "   from cli arg: ";
    msg += s;
    // include the reason
    msg += "   err: ";
    msg += reason;
    throw std::runtime_error(msg);
  }
};

/*! Helper function used to handle some parsing-related tasks that come up when considering the
 *  full name of a parameter-table or a parameter-key.
 *
 *  This does the following:
 *    1. Validates the full_name only contains allowed characters
 *    2. for a name "a.b.c.d", we step through the "a.b.c", "a.b", and "a" to
 *       (i)   confirm that no-segment is empty
 *       (ii)  ensure that the part is registered as a table
 *       (iii) ensure that the part does not collide with the name of a parameter
 *
 *  \returns An empty string if there aren't any problems. Otherwise the returned string provides an error message
 */
std::string Process_Full_Name(std::string full_name, std::map<std::string, bool, std::less<>>& full_table_map,
                              const std::map<std::string, ParameterMap::ParamEntry>& param_entries)
{
  // first, confirm the name only holds valid characters
  std::size_t bad_value_count = 0;
  for (char ch : full_name) {
    bad_value_count += ((ch != '.') and (ch != '_') and (ch != '-') and not std::isalnum(ch));
  }
  if (bad_value_count > 0) {
    return "contains an unallowed character";
  }

  // now lets step through the parts of a name (delimited by the '.')
  // -> for a name "a.b.c.d", we check "a.b.c", "a.b", and "a"
  // -> specifically, we (i)   confirm that no-segment is empty
  //                     (ii)  ensure that the table is registered
  //                     (iii) ensure there aren't any collisions with parameter-names
  const std::size_t size_minus_1 = full_name.size() - 1;
  std::size_t rfind_start        = size_minus_1;
  while (true) {
    const std::size_t pos = full_name.rfind('.', rfind_start);
    if (pos == std::string_view::npos) return {};
    if (pos == size_minus_1) return "ends with a '.' character";
    if (pos == 0) return "start with a '.' character";
    if (pos == rfind_start) return "contains contiguous '.' characters";

    std::string_view table_name_prefix(full_name.data(), pos);

    // if table_name_prefix has been seen before, then we're done (its parents have been seen too)
    if (full_table_map.find(table_name_prefix) != full_table_map.end()) return {};

    // register table_name_prefix for the future
    std::string table_name_prefix_str(table_name_prefix);
    full_table_map[table_name_prefix_str] = false;  // we assign false because it was implicitly declared

    if (param_entries.find(table_name_prefix_str) != param_entries.end()) {
      return "the (sub)table name collides with the existing \"" + table_name_prefix_str + "\" parameter";
    }

    rfind_start = pos - 1;
  }
}

}  // anonymous namespace

ParameterMap::ParameterMap(std::FILE* fp, int n_param_override_args, char** param_override_args, bool close_fp)
{
  CHOLLA_ASSERT(fp != nullptr, "ParameterMap was passed a nullptr rather than an actual file object");
  FileLineStream file_line_stream(fp);
  CliLineStream cli_line_stream(n_param_override_args, param_override_args);

  // to provide consistent table-related behavior to TOML, we need to track the names of tables to ensure that:
  //  1. no table is explicitly defined more than once
  //  2. no table name collides with a parameter name.
  //
  // To accomplish this, we:
  //  -> we add all explicitly defined table-names (e.g. we add "my-table" if we encounter the [my-table] header).
  //     The associated value is true if it was explicitly defined
  //  -> we also tracks implicitly defined tables. An implicitly defined table is associated with a false value.
  //     A table can be implicitly defined:
  //     1. In an explicit definition. For example, the [my.first.table] header implicitly defines the table
  //        "my.first.table". It also implicitly defines the "my.first" and "my" tables when they don't exist
  //     2. In a dotted parameter name. For example `my.table.val=3` defines the `my.table.val` parameter. It also
  //        implicitly defines the `my.table
  //  -> std::less<> is here so we can compare perform "heterogenous-lookups." In other words, it lets us check if
  //     the set contains a string specified in a `std::string_view` instance even though the set internally stores
  //     `std::string` instances (without heterogenous-lookups we couldn't use std::string_view)
  std::map<std::string, bool, std::less<>> all_tables;

  std::string cur_table_header{};

  /* Read next line */
  while (file_line_stream.next()) {
    char* buff = file_line_stream.s;

    /* Skip blank lines and comments */
    if (buff[0] == '\n' || buff[0] == '#' || buff[0] == ';') {
      continue;
    }

    if (buff[0] == '[') {  // here we are parsing a header like "[my_table]\n"
      std::string_view view(buff);
      rstrip(view);  // strip off trailing whitespace from the view
      if (view.back() != ']') file_line_stream.error("problem parsing a parameter-table header");
      cur_table_header = view.substr(1, view.size() - 2);
      if (cur_table_header.size() == 0) file_line_stream.error("empty table-names aren't allowed");

      // confirm that we haven't seen this header before (and that there isn't a parameter with the same name)
      auto search = all_tables.find(cur_table_header);
      if ((search != all_tables.end()) and (search->second)) {
        // when search->second is false, there's no issue (since the table was never explicitly defined)
        file_line_stream.error("the same table header can't appear more than once", cur_table_header, true);
      } else if (this->entries_.find(cur_table_header) != this->entries_.end()) {
        file_line_stream.error("table-name collides with a parameter of the same name");
      }

      std::string msg = Process_Full_Name(cur_table_header, all_tables, this->entries_);
      if (not msg.empty()) file_line_stream.error(msg, cur_table_header, true);

      // record that we've seen this header (for future checks)
      all_tables[cur_table_header] = true;

    } else {  // Parse name/value pair from line
      KeyValueExtraction kv_pair = Try_Extract_Key_Value(buff);
      if (not kv_pair.maybe_value.has_value()) {
        // file_line_stream.warn("skipping line due to invalid format (this may become an error in the future)");
        // continue;
        file_line_stream.error(kv_pair.error_msg);
      } else if (kv_pair.n_key_segments != 1) {
        file_line_stream.error("parameter-names in the parameter-file aren't currently allowed to contain a '.'");
      } else {
        std::string full_param_name = (not cur_table_header.empty()) ? (cur_table_header + '.') : std::string{};
        full_param_name += kv_pair.key;
        param_details::Value& value = kv_pair.maybe_value.value();

        std::string msg = Process_Full_Name(full_param_name, all_tables, this->entries_);
        if (not msg.empty()) file_line_stream.error(msg, full_param_name, false);
        entries_[full_param_name] = {value, false};
      }
    }
  }

  // Parse overriding args from command line
  while (cli_line_stream.next()) {
    // try to parse the argument
    KeyValueExtraction kv_pair = Try_Extract_Key_Value(cli_line_stream.s);
    if (not kv_pair.maybe_value.has_value()) {
      cli_line_stream.error(kv_pair.error_msg);
    } else {
      std::string key_str(kv_pair.key);
      param_details::Value& value = kv_pair.maybe_value.value();
      std::string msg             = Process_Full_Name(key_str, all_tables, this->entries_);
      if (not msg.empty()) cli_line_stream.error(msg, key_str, false);
      std::string value_str_ = value.toml_repr();
      chprintf("Override with %s=%s\n", key_str.c_str(), value_str_.c_str());
      entries_[key_str] = {value, false};
    }
  }

  if (close_fp) std::fclose(fp);
}

/*! try to open a file. */
static std::FILE* open_file_(const std::string& fname)
{
  std::FILE* fp = std::fopen(fname.c_str(), "r");
  if (fp == nullptr) CHOLLA_ERROR("failed to read parameter file: %s", fname.c_str());
  return fp;
}

ParameterMap::ParameterMap(const std::string& fname, int argc, char** argv)
    : ParameterMap(open_file_(fname), argc, argv, true)
{
}

int ParameterMap::warn_unused_parameters(const std::set<std::string>& ignore_params, bool abort_on_warning,
                                         bool suppress_warning_msg) const
{
  int unused_params = 0;
  for (const auto& kv_pair : entries_) {
    const std::string& name                     = kv_pair.first;
    const ParameterMap::ParamEntry& param_entry = kv_pair.second;

    if ((not param_entry.accessed) and (ignore_params.find(name) == ignore_params.end())) {
      unused_params++;
      const std::string& value = param_entry.val.toml_repr();
      if (abort_on_warning) {
        CHOLLA_ERROR("%s/%s:  Unknown parameter/value pair!", name.c_str(), value.c_str());
      } else if (not suppress_warning_msg) {
        chprintf("WARNING: %s/%s:  Unknown parameter/value pair!\n", name.c_str(), value.c_str());
      }
    }
  }
  return unused_params;
}

void ParameterMap::Enforce_Table_Content_Uniform_Access_Status(std::string table_name, bool expect_unused) const
{
  // error check:
  std::string_view table_name_view(table_name);
  if (table_name_view.back() == '.') table_name_view = table_name_view.substr(0, table_name_view.size() - 1);
  CHOLLA_ASSERT(table_name_view.size() > 0, "the table_name_view must contain at least one character");

  std::string prefix      = std::string(table_name_view) + '.';
  std::size_t prefix_size = prefix.size();

  std::string problematic_parameter{};
  for (auto it = entries_.lower_bound(prefix); it != entries_.end(); ++it) {
    const std::string& name                     = it->first;
    const ParameterMap::ParamEntry& param_entry = it->second;
    if (name.compare(0, prefix_size, prefix) != 0) break;

    if (param_entry.accessed == expect_unused) {
      problematic_parameter = name;
      break;
    }
  }

  if (problematic_parameter.size() == 0) return;  // no issues!

  // Report the errors:
  if (expect_unused) {
    CHOLLA_ERROR("Internal Error: the %s shouldn't have been accessed yet", problematic_parameter.c_str());
  } else {
    // gather the parameters that have been accessed (for an informative message)
    std::string par_list{};
    for (auto it = entries_.lower_bound(prefix); it != entries_.end(); ++it) {
      const std::string& name                     = it->first;
      const ParameterMap::ParamEntry& param_entry = it->second;
      if (name.compare(0, prefix_size, prefix) != 0) break;

      if (param_entry.accessed) {
        par_list += "\n   ";
        par_list += name;
      }
    }

    if (par_list.size() > 0) {
      CHOLLA_ERROR("Based on the parameter(s):%s\nthe %s parameter should not be present in the parameter file",
                   par_list.c_str(), problematic_parameter.c_str());
    }
    CHOLLA_ERROR("Something is wrong, the %s parameter should not be present in the parameter file",
                 problematic_parameter.c_str());
  }
}
