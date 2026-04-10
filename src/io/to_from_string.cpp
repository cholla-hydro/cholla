#include "to_from_string.h"

#include <algorithm>  // std::min
#include <optional>
#include <string_view>
#include <utility>

#include "../utils/error_handling.h"

namespace io
{

/*! return the number of contiguous bare key characters in \p s (starting from
 *  position \p pos )
 *
 *  \param s The string segment to attempt to parse
 *  \param pos The index to start counting from
 *
 *  A bare TOML key only consists of ASCII characters that are letters, digits,
 *  underscores or dashes. These do not need to be quoted. More details can be found is
 *  defined at https://toml.io/en/v1.1.0#keys
 */
static std::size_t count_contig_bare_key_chars(std::string_view s, std::size_t pos = 0)
{
  const std::size_t size = s.size();
  if (pos > size) return pos;

  for (std::size_t i = pos; i < size; i++) {
    char chr = s[i];
    // if this ends up being slow, we should look into replacing `or` & `and` with the
    // bitwise analogues since they don't short-circuit
    bool is_digit             = ('0' <= chr) and (chr <= '9');
    bool is_upper             = ('A' <= chr) and (chr <= 'Z');
    bool is_lower             = ('a' <= chr) and (chr <= 'z');
    bool is_underscore_hyphen = ('_' == chr) or ('-' == chr);

    bool is_valid = is_digit or is_upper or is_lower or is_underscore_hyphen;
    if (not is_valid) return i;
  }

  return size;
}

using namespace std::literals;
constexpr inline std::string_view SPACE_OR_TAB = " \t"sv;

// the equivalent regex is "[ \\t]*\\.[ \\t]*"
static std::size_t match_dot_sep_(std::string_view s, std::size_t pos = 0)
{
  // check for leading whitespace
  std::size_t tmp = s.find_first_not_of(SPACE_OR_TAB, pos);
  if (tmp != std::string_view::npos) pos = tmp;
  // check for the dot
  if (s[pos] != '.') {
    return std::string_view::npos;
  }
  pos++;
  // check for trailing whitespace
  tmp = s.find_first_not_of(SPACE_OR_TAB, pos);
  if (tmp != std::string_view::npos) pos = tmp;
  return pos;
}

KeyParseRslt try_parse_param_key(std::string_view s, std::size_t pos)
{
  std::size_t size = s.size();
  if (pos > size) CHOLLA_ERROR("pos can't exceed s.size()");

  auto err = [](std::string msg) -> KeyParseRslt { return {0, msg, 0}; };

  std::string out;
  int n_segments = 0;
  while (true) {
    std::size_t n_chars = count_contig_bare_key_chars(s, pos);
    if (n_chars == 0) {
      bool end_of_s = pos == size;
      char invalid  = (end_of_s) ? '\0' : s[pos];
      if (n_segments == 0) {
        if (end_of_s) return err("no characters to parse");
        if (invalid == '.') return err("can't start with a dot");
        return err(std::string("invalid character: '") + invalid + std::string("'"));
      } else {
        if (end_of_s) return err("can't end with '.'");
        if (invalid == '.') return err("can't contain 2 '.' characters in a row");
        return err(std::string("can't end with '.' (the next character, '") + invalid +
                   std::string("', isn't a valid key character in a bare key)"));
      }
    }

    out.append(s, pos, n_chars);
    n_segments++;
    pos += n_chars;

    std::size_t tmp = match_dot_sep_(s, pos);
    if (tmp == std::string_view::npos) {
      return {pos, out, n_segments};
    }
    out.push_back('.');
    pos = tmp;
  }
}

// this lists all compact escape sequences known to TOML
// (for now, we ignore the sequence for escaping arbitrary unicode characters)
static constexpr std::pair<char, char> backslash_escape_pairs_[]{
    {'b', '\b'},    // <- backspace
    {'t', '\t'},    // <- tab
    {'n', '\n'},    // <- linefeed
    {'f', '\f'},    // <- form feed
    {'r', '\r'},    // <- carriage return
    {'e', '\x1b'},  // <- escape ('\e' isn't in C++ standard, but is understood by gcc)
    {'"', '"'},     // <- quote
    {'\\', '\\'},   // <- backslash
};

std::optional<char> Lookup_From_Post_Backslash_(char c)
{
  for (const std::pair<char, char>& pair : backslash_escape_pairs_) {
    if (c == pair.first) return {pair.second};
  }
  return std::nullopt;
}

// construct a std::string holding the properly toml-encoding of the specified
// '\0'-terminated string, `s`.
//
// NOTE: this certainly isn't the fastest implementation, but my hope is that we'll
//       eventually start using something like toml++
std::string encode_toml_str(std::string_view s)
{
  // check if we need to escape any character
  std::size_t size                = s.size();
  std::size_t n_requires_escaping = 0;
  for (std::size_t i = 0; i < size; i++) {
    char chr = s[i];
    for (const std::pair<char, char>& pair : backslash_escape_pairs_) {
      n_requires_escaping += (chr == pair.second);
    }
  }

  std::string out;
  out.reserve(2 + n_requires_escaping + size);
  out.push_back('"');

  if (size == 0) {
    // do nothing
  } else if (n_requires_escaping == 0) {
    out.append(s);
  } else {
    for (std::size_t i = 0; i < size; i++) {
      char chr                = s[i];
      char post_backslash_chr = '\0';
      for (const std::pair<char, char>& pair : backslash_escape_pairs_) {
        if (chr == pair.second) {
          post_backslash_chr = pair.first;
          break;
        }
      }
      if (post_backslash_chr == '\0') {
        out.push_back(chr);
      } else {
        out.push_back('\\');
        out.push_back(post_backslash_chr);
      }
    }
  }
  out.push_back('"');
  return out;
}

std::string encode_toml_key(std::string_view key)
{
  std::size_t tmp     = count_contig_bare_key_chars(key);
  bool allow_bare_key = (tmp == key.size()) and (tmp > 0);
  return allow_bare_key ? std::string(key) : encode_toml_str(key);
}

/*! aborts with an error if c isn't allowed in a raw-string/basic-string */
[[noreturn]] static void Abort_Bad_Char_Err_(char c, const char* string_kind)
{
  // reminder: the C++ standard doesn't define whether char is signed or not (if it did,
  // we'd only need to check one of the inequalities for determining if its ascii)
  bool is_ascii = (c < 0) || (c > 127);
  if (is_ascii) CHOLLA_ERROR("encountered invalid contol-code in %s", string_kind);
  CHOLLA_ERROR("encounter a unicode character (not supported) or invalid byte in %s", string_kind);
}

/*! return if the argument is an ASCII character that can be specified within a
 *  "raw-string" or "basic-string" without escaping.
 *
 *  Reminder: see the docstring of \ref try_parse_param_str for more about raw-strings
 *  and basic-strings
 */
static bool Is_Common_Unescaped_Ascii_(char c)
{
  bool is_tab = c == '\t';

  // all other unescaped characters are in the range `(32 <= c) && (c < 127)`
  bool in_range = (c >= 32) and (c < 127);
  // the only exceptions are:
  bool not_apostrophe = (c != '\'');  // <- this char can't appear in raw-strings
  bool not_quote      = (c != '"');   // <- this char requires escaping in basic-strings
  bool not_backslash  = (c != '\\');  // <- this char requires escaping in basic-strings

  return is_tab or (in_range and not_apostrophe and not_quote and not_backslash);
}

std::pair<std::size_t, std::string> try_parse_param_str(std::string_view s, std::size_t pos)
{
  // this is the result that we will return if s isn't a properly encoded string
  std::pair<std::size_t, std::string> not_a_string_rslt{pos, std::string()};

  std::size_t size = s.size();
  bool nchr_geq_2  = (size >= pos + 2);
  bool nchr_geq_3  = (size >= pos + 3);

  if (nchr_geq_3 and s[pos] == '\'' and s[pos + 1] == '\'' and s[pos + 2] == '\'') {
    CHOLLA_ERROR("no support for parsing a multiline raw-string");

  } else if (nchr_geq_3 and s[pos] == '"' and s[pos + 1] == '"' and s[pos + 2] == '"') {
    CHOLLA_ERROR("no support for parsing a multiline basic-string");

  } else if (nchr_geq_2 and s[pos] == '\'') {  // <- raw-string
    pos++;
    std::string out;  // <- initialized to empty string
    while (pos < size) {
      char c = s[pos];

      if (c == '\'') {
        return {pos + 1, out};
      } else if (Is_Common_Unescaped_Ascii_(c) or (c == '"') or (c == '\\')) {
        out.push_back(c);
      } else {
        Abort_Bad_Char_Err_(c, "raw-string");
      }
      pos++;
    }
    CHOLLA_ERROR("the raw-string doesn't have a closing '");

  } else if (nchr_geq_2 and s[pos] == '"') {  // <- basic-string
    pos++;
    std::string out;  // <- initialized to empty string
    while (pos < size) {
      char c = s[pos];

      if (c == '"') {
        return {pos + 1, out};
      } else if (Is_Common_Unescaped_Ascii_(c) or (c == '\'')) {
        out.push_back(c);
      } else if (c == '\\') {          // <- handle backslash escaping
        if ((pos + 1) == size) break;  // <- this will trigger the appropriate error
        pos++;
        c = s[pos];
        if (c == 'x' or c == 'u' or c == 'U') {
          CHOLLA_ERROR("no support for unicode escape sequences starting with \\%c", c);
        }
        std::optional<char> tmp = Lookup_From_Post_Backslash_(c);
        if (tmp.has_value()) {
          out.push_back(tmp.value());
        } else {
          CHOLLA_ERROR("no support for unicode escape sequences starting with \\%c", c);
        }
      } else {
        Abort_Bad_Char_Err_(c, "basic-string");
      }
      pos++;
    }
    CHOLLA_ERROR("the basic-string doesn't have a closing \"");

  } else {
    return not_a_string_rslt;
  }
}

}  // namespace io