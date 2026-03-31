#include "AttrRecorderInterface.h"

#include <string_view>
#include <utility>

namespace io_detail
{

// this lists all compact escape sequences known to TOML
// (for now, we ignore the sequence for escaping arbitrary unicode characters)
static constexpr std::pair<char, char> backslash_escape_pairs_[]{
    {'b', '\b'},   // <- backspace
    {'t', '\t'},   // <- tab
    {'n', '\n'},   // <- linefeed
    {'f', '\f'},   // <- form feed
    {'r', '\r'},   // <- carriage return
    {'e', '\e'},   // <- escape
    {'"', '"'},    // <- quote
    {'\\', '\\'},  // <- backslash
};

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
  const std::size_t size = key.size();

  // determine whether we can write a bare key (i.e. without enclosing quotes)
  // -> we can do this if each character matches one of A-Za-z0-9_-
  //    https://toml.io/en/v1.1.0#keys
  bool allow_bare_key = size > 0;  // <- if the key has zero characters it needs quotes
  for (std::size_t i = 0; i < size; i++) {
    char chr = key[i];
    // if this ends up being slow, we should look into replacing `or` & `and` with the
    // bitwise analogues since they don't short-circuit
    bool is_digit             = ('0' <= chr) and (chr <= '9');
    bool is_upper             = ('A' <= chr) and (chr <= 'Z');
    bool is_lower             = ('a' <= chr) and (chr <= 'z');
    bool is_underscore_hyphen = ('_' == chr) or ('-' == chr);

    if (is_digit or is_upper or is_lower or is_underscore_hyphen) {
      continue;
    } else {
      allow_bare_key = false;
    }
  }

  return allow_bare_key ? std::string(key) : encode_toml_str(key);
}

}  // namespace io_detail