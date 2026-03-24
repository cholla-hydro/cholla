/*!
 * \file FnameTemplate.h
 * \brief Declares the FnameTemplate type
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "../io/ParameterMap.h"
#include "../utils/error_handling.h"

/*! Lightweight object designed to centralize the file-naming logic (& any associated configuration).
 *
 * Cholla pathnames traditionally followed the following template:
 *     "{outdir}{nfile}{pre_extension_suffix}{extension}.{proc_id}"
 * where each curly-braced token represents a different variable. In detail:
 *   - `{outdir}` is the parameter from the parameter file. The historical behavior (that we currently
 *     maintain), if this is non-empty, then all charaters following the last '/' are treated as a
 *     prefix to the output file name (if there aren't any '/' characters, then the whole string is
 *     effectively a prefix.
 *   - `{nfile}` is the current file-output count.
 *   - `{pre_extension_suffix}` is the pre-hdf5-extension suffix. It's the suffix that precedes the
 *     file extension (or `{extension}`)
 *   - `{extension}` is the filename extension. Examples include ".h5" or ".bin" or ".txt".
 *   - `{proc_id}` represents the process-id that held the data that will be written to this file.
 *     Previously, in non-MPI runs, this was omitted.
 *
 * Instances can be configured to support the following newer file-naming template
 *    "{outdir}/{nfile}/{nfile}{pre_extension_suffix}{extension}.{proc_id}"
 * where the the significance of each curly-braced token is largely unchanged. There are 2 things
 * worth noting:
 *   - all files written at a single simulation-cycle are now grouped in a single directory
 *   - `{outdir}` never specifies a file prefix. When `{outdir}` is empty, it is treated as "./".
 *     Otherwise, we effectively append '/' to the end of `{outdir}`
 *
 * \note
 * This could probably pull double-duty and get reused with infile.
 */
class FnameTemplate
{
 public:
  FnameTemplate() = delete;

  FnameTemplate(bool separate_cycle_dirs, std::string outdir)
      : separate_cycle_dirs_(separate_cycle_dirs), outdir_(std::move(outdir))
  {
  }

  static FnameTemplate from_pmap(ParameterMap& pmap)
  {
    bool legacy_flat_outdir;
    int uncoerced = pmap.value_or("legacy_flat_outdir", 0);
    if (uncoerced == 0) {
      legacy_flat_outdir = false;
    } else if (uncoerced == 1) {
      legacy_flat_outdir = true;
    } else {
      CHOLLA_ERROR("legacy_flat_outdir parameter must be 1 or 0.");
    }
    return FnameTemplate(not legacy_flat_outdir, pmap.value_or("outdir", ""));
  }

  /*! Specifies whether separate cycles are written to separate directories */
  bool separate_cycle_dirs() const noexcept { return separate_cycle_dirs_; }

  /*! Returns the nominal output-directory (this value is unaffected by separate_cycle_dirs()) */
  std::string nominal_output_dir_path() const noexcept { return outdir_; }

  /*! Returns the effective output-directory used for outputs at a given simulation-cycle */
  std::string effective_output_dir_path(int nfile) const noexcept;

  /**\{*/  // <- the functions inside this doxygen group share a docstring
  /*! format the file path */
  std::string format_fname(int nfile, std::string_view pre_extension_suffix,
                           std::optional<std::string_view> post_extension_suffix = std::nullopt) const noexcept;

  std::string format_fname(int nfile, int file_proc_id, std::string_view pre_extension_suffix,
                           std::optional<std::string_view> post_extension_suffix = std::nullopt) const noexcept;
  /**\}*/

 private:
  bool separate_cycle_dirs_;
  std::string outdir_;
};
