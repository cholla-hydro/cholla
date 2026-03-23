#include "AttrRecorderInterface.h"

#include <cstring>
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

// write out the '\0'-terminated string, `s`, to a toml file.
//
// This variant of the function is passed the length, ``len``. It excludes the trailing
// '\0' (but it is definitely present)
//
// NOTE: this certainly isn't the fastest implementation, but my hope is that we'll
//       eventually start using something like toml++
static void write_toml_str_(const char* s, std::size_t len, std::FILE* fp)
{
  std::fputc('"', fp);

  // check if we need to escape any character
  std::size_t n_requires_escaping = 0;
  for (std::size_t i = 0; i < len; i++) {
    char chr = s[i];
    for (const std::pair<char, char>& pair : backslash_escape_pairs_) {
      n_requires_escaping += (chr == pair.second);
    }
  }

  if (len == 0) {
    // do nothing
  } else if (n_requires_escaping == 0) {
    std::fputs(s, fp);
  } else {
    for (std::size_t i = 0; i < len; i++) {
      char chr                = s[i];
      char post_backslash_chr = '\0';
      for (const std::pair<char, char>& pair : backslash_escape_pairs_) {
        if (chr == pair.second) {
          post_backslash_chr = pair.first;
          break;
        }
      }
      if (post_backslash_chr == '\0') {
        std::fputc(chr, fp);
      } else {
        std::fputc('\\', fp);
        std::fputc(post_backslash_chr, fp);
      }
    }
  }
  std::fputc('"', fp);
}

void write_toml_str(const char* s, std::FILE* fp) { write_toml_str_(s, std::strlen(s), fp); }

/*! Write the toml key to file (performing any escaping if necessary) */
void write_toml_key(const char* key, std::FILE* fp)
{
  // get the size (excluding the trailing '\0')
  const std::size_t size = std::strlen(key);

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

  if (allow_bare_key) {
    std::fputs(key, fp);
  } else {
    write_toml_str_(key, size, fp);
  }
}

}  // namespace io_detail