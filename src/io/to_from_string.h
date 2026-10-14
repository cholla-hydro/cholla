/*! \file
 *  \brief Define misc functions for converting to/from string values
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace io
{

/*! Construct the proper toml representation of the specified key
 *
 *  This performs any necessary escaping.
 */
std::string encode_toml_key(std::string_view key);

/*! Construct the proper toml representation of the specified string
 *
 *  This performs any necessary escaping.
 */
std::string encode_toml_str(std::string_view s);

/*! Extract the string from segment of the parameter file
 *
 *  \param s The segment of the parameter file to attempt to parse
 *  \param pos Position at which to attempt to start parsing
 *
 *  Essentially, we support a TOML "basic-string" of ascii characters or a TOML
 *  "raw-string" of ascii characters. For simplicity, we explicitly disallow unicode
 *  characters and multi-line strings.
 *
 *  While we refer the reader to https://toml.io/en/v1.1.0#string for formal
 *  definitions, we provide quick high-level descriptions below:
 *  - a "basic-string" is double quoted & acts like a string-literal in python
 *    (e.g. "hello world" or "hi\nworld")
 *  - a "raw-string" is single-quoted and mostly acts mostly like a raw string-literal
 *    in python (e.g. r'hi world'). Unlike the python counterpart, quotes are not
 *    escaped by a backslash (e.g. we accept '\' as a valid "raw-string").
 *
 *
 *  \returns A pair holding the position of the first character after the end of the
 *      and the parsed value. If the first element is 0, then there was a parsing
 *      issue.
 */
std::pair<std::size_t, std::string> try_parse_param_str(std::string_view s, std::size_t pos = 0);

}  // namespace io