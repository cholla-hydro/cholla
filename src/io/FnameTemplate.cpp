/*!
 * \file FnameTemplate.cpp
 * \brief Implements the FnameTemplate type
 */
#include "../io/FnameTemplate.h"

#include <filesystem>
#include <sstream>
#include <string>

std::string FnameTemplate::effective_output_dir_path(int nfile) const noexcept
{
  // for consistency, ensure that the returned string always has a trailing "/"
  if (outdir_.empty()) {
    return "./";
  } else if (separate_cycle_dirs_) {
    return this->outdir_ + "/" + std::to_string(nfile) + "/";
  } else {
    // if the last character of outdir is not a '/', then the substring of
    // characters after the final '/' (or entire string if there isn't any '/')
    // is treated as a file-prefix
    //
    // this is accomplished here:
    std::filesystem::path without_file_prefix = std::filesystem::path(this->outdir_).parent_path();
    return without_file_prefix.string() + "/";
  }
}

std::string FnameTemplate::format_fname(int nfile, std::string_view pre_extension_suffix,
                                        std::optional<std::string_view> post_extension_suffix) const noexcept
{
#ifdef MPI_CHOLLA
  int file_proc_id = procID;
#else
  int file_proc_id = 0;
#endif
  return format_fname(nfile, file_proc_id, pre_extension_suffix, post_extension_suffix);
}

std::string FnameTemplate::format_fname(int nfile, int file_proc_id, std::string_view pre_extension_suffix,
                                        std::optional<std::string_view> post_extension_suffix) const noexcept
{
  // get the leading section of the string
  const std::string path_prefix =
      (separate_cycle_dirs_)
          ? (effective_output_dir_path(nfile) + "/")  // while redundant, the slash signals our intent
          : outdir_;

  // get the file extension
#ifdef HDF5
  const char *extension = ".h5";
#else
  const char *extension = ".txt";
#endif

  std::stringstream s;

  s << path_prefix << std::to_string(nfile) << pre_extension_suffix << extension;
  if (post_extension_suffix.has_value()) {
    s << '.' << *post_extension_suffix;
  }
  s << '.' << std::to_string(file_proc_id);
  return s.str();
}
