/*! \file
 *  \brief Define misc functions for converting to/from string values
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "../utils/error_handling.h"

namespace io
{

struct KeyParseRslt {
  std::size_t pos;
  std::string val;
  int n_segments;
};

/*! Try to extract a parameter key from \p s
 *
 *  \param s The string segment to attempt to parse
 *  \param pos Position at which to attempt to start parsing
 *
 *  At this, all parameter keys understood by Cholla are valid TOML keys, but not all
 *  TOML keys are valid. We currently restrict parameter keys to be either:
 *  - a "bare key"
 *  - a "dotted key", where each segment in the sequence is a "bare key".
 *
 *  At this time, we expressly forbid TOML's quoted keys. We refer the reader to
 *  https://toml.io/en/v1.1.0#string for formal definitions.
 *
 *  \note
 *  At this point, parsing TOML's "quoted key" is easy (we can just use
 *  \p try_parse_param_str ). But, that involves changing how \ref ParameterMap
 *  tracks keys.
 */
KeyParseRslt try_parse_param_key(std::string_view s, std::size_t pos = 0);

/*! Construct the proper toml representation of the specified key.
 *
 *  For some added context,
 *  This performs any necessary escaping. We assume that occurences of the '.'
 *  character should be escaped.
 */
std::string encode_toml_key(std::string_view undotted_key);

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