#ifndef S99TABLE_H
#define S99TABLE_H

#include <string>
#include <vector>

#include "../utils/error_handling.h"

namespace feedback{

/* kinds of starburst99 datatables used by cholla */
enum class S99TabKind { supernova, stellar_wind };

/* Class that represents a parsed starburst99 table. This is a little over the top, but it's
 * probably fine. We just use this when reading the data out of the table 
 */
class S99Table {

public:
  S99Table(std::vector<std::string> col_names, std::vector<double> data,
           std::size_t ncols, std::size_t nrows)
    : col_names_(col_names), data_(data), ncols_(ncols), nrows_(nrows)
  { }

  /* number of rows in the table */
  std::size_t nrows() const noexcept {return nrows_;}

  /* number of columns in the table */
  std::size_t ncols() const noexcept {return ncols_;}

  /* access an entry in the table */
  double operator() (std::size_t col_ind, std::size_t row_ind) const noexcept
  {
    // probably could remove this check...
    if ((col_ind >= ncols_) or (row_ind >= nrows_)) {
      CHOLLA_ERROR("invalid index col_ind, %zu, must be less than %zu and row_ind, %zu, must be "
                   "less than %zu",
                   col_ind, ncols_, row_ind, nrows_);
    }
    return data_[row_ind * ncols_ + col_ind];
  }

  /* query the name of a given column */
  std::string col_name(std::size_t col_ind) const noexcept {
    if (col_ind < ncols_) return col_names_[col_ind];
    return "";
  }

  /* Returns the index of the specified column. */
  std::size_t col_index(const std::string& col_name) const noexcept
  {
    for (std::size_t i = 0; i < ncols_; i++) {
      if (col_names_[i] == col_name) return i;
    }
    CHOLLA_ERROR("the table doesn't hold a column called: \"%s\"", col_name.c_str());
  }

private:  // attributes  
  std::vector<std::string> col_names_;
  std::vector<double> data_;
  std::size_t ncols_;
  std::size_t nrows_;
};

}

/* Parse a Starburst99 table 
 *
 * TODO: put this back into the feedback namespace
 */
feedback::S99Table parse_s99_table(const std::string& fname, feedback::S99TabKind kind);

#endif /* S99TABLE_H */