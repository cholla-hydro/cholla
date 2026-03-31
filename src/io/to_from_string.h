/*! \file
 *  \brief Define misc functions for converting to/from string values
 */

#pragma once

#include <string>
#include <string_view>

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
}  // namespace io