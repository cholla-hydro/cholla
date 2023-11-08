#include "s99table.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <string_view>

namespace { // anonymous namespace

std::string_view rstripped_view_(std::string_view s) {
  std::size_t i = s.size();
  while (i > 0) {
    i--;
    if (s[i] != ' ')  return s.substr(0, i+1);
  }
  return {};
}

std::string_view trimmed_view_(std::string_view s) {
  std::size_t substr_start = s.find_first_not_of(' ');
  if (substr_start == std::string_view::npos) return {};
  std::string_view lstripped_view = s.substr(substr_start);
  return rstripped_view_(lstripped_view);
}

/* counts the number of contiguous ' ' characters at start of a string that is
 * null-character-terminated */
std::size_t leading_space_count_(const char* s) {
  for(std::size_t i = 0; true; i++) {
    if (s[i] != ' ') return i;
  }
}

template<std::size_t min_contig_space_count>
struct TokenFinder {

  const std::string& line;
  std::size_t next_token_start = 0; // first index of next token

  /* gets the next token. The template argument specifies the minimum number of
   * spaces required to delimit tokens (relevant for parsing col names) */
  std::string_view next() {
    std::size_t start = (next_token_start == 0)
      ? line.find_first_not_of(' ') : next_token_start;
    if (line.size() == start) return std::string_view();

    for (std::size_t i = start; i < line.size(); /* update i within loop */) {
      const std::size_t space_count = leading_space_count_(line.data() + i);
      if (space_count >= min_contig_space_count) {
        this->next_token_start = space_count + i;
        return std::string_view(line.data() + start, i - start);
      }
      i += (space_count > 0) ? space_count : 1;
    }

    this->next_token_start = line.size();
    return std::string_view(line.data() + start, line.size() - start);
  }

};


std::string descr_string_(feedback::S99TabKind kind) {
  switch(kind) {
    case feedback::S99TabKind::supernova:
      return "RESULTS FOR THE SUPERNOVA RATE";
    case feedback::S99TabKind::stellar_wind:
      return "RESULTS FOR THE **STELLAR** WIND POWER AND ENERGY";
    default:
      CHOLLA_ERROR("there is an unhandled s99 table kind");
  }
}

std::vector<std::string> better_col_names(std::vector<std::string> parsed_names,
                                          feedback::S99TabKind kind) {
  std::string kind_name;
  std::vector<std::string> full_names{};
  switch(kind) {
    case feedback::S99TabKind::supernova:
      kind_name = "supernova";
      full_names = {
        "TIME",
        "ALL SUPERNOVAE: TOTAL RATE", "ALL SUPERNOVAE: POWER",
        "ALL SUPERNOVAE: ENERGY",
        "TYPE IB SUPERNOVAE: TOTAL RATE", "TYPE IB SUPERNOVAE: POWER",
        "TYPE IB SUPERNOVAE: ENERGY",
        "ALL SUPERNOVAE: TYPICAL MASS", "ALL SUPERNOVAE: LOWEST PROG. MASS",
        "STARS + SUPERNOVAE: POWER", "STARS + SUPERNOVAE: ENERGY"
      };
      break;
    case feedback::S99TabKind::stellar_wind:
      kind_name = "stellar wind";
      full_names = {
        "TIME",
        "POWER: ALL", "POWER: OB", "POWER: RSG", "POWER: LBV", "POWER: WR",
        "ENERGY: ALL",
        "MOMENTUM FLUX: ALL", "MOMENTUM FLUX: OB", "MOMENTUM FLUX: RSG",
        "MOMENTUM FLUX: LBV", "MOMENTUM FLUX: WR"
      };
      break;
    default:
      CHOLLA_ERROR("there is an unhandled kind of starburst99 table");
  }

  if (full_names.size() != parsed_names.size()) {
    CHOLLA_ERROR("A \"%s\" starburst99 table should have %zu cols, not %zu",
                 kind_name.c_str(), full_names.size(), parsed_names.size());
  }

  for (std::size_t i = 0; i < parsed_names.size(); i++) {
    const std::string& full_name = full_names[i];

    // parsed_names should match everything after the ": " substring if its
    // present (if the ": " substring isn't present, match the full string)
    std::size_t separator_pos = full_name.find(": ");
    std::string expected_name;
    if (separator_pos == std::string::npos) {
      expected_name = full_name;
    } else {
      expected_name = full_name.substr(separator_pos+2);
    }

    if (expected_name != parsed_names[i]) {
      CHOLLA_ERROR("column %zu is expected to be labelled %s in a \"%s\" "
                   "starburst99 table, not \"%s\"",
                   i, expected_name.c_str(), kind_name.c_str(),
                   parsed_names[i].c_str());
    }
  }
  return full_names;
}

} // close anonymous namespace

feedback::S99Table parse_s99_table(const std::string& fname, feedback::S99TabKind kind)
{
  // read in the file
  std::ifstream stream{fname};
  if (!stream.is_open()) {
    CHOLLA_ERROR("problem opening %s", fname.c_str());
  }

  // parse the first 6 lines of the header
  for (int i = 0; i < 6; i++) {
    std::string line;
    std::getline(stream, line);
    if (i == 3) {
      std::string expected_description_str = descr_string_(kind);
      if (trimmed_view_(line) != expected_description_str) {
        std::string trimmed_line = std::string(trimmed_view_(line));
        CHOLLA_ERROR("Expected the description string: \"%s\" on the fourth "
                     "line of the table. Instead, the line reads: \"%s\"",
                     expected_description_str.c_str(), trimmed_line.c_str());
      }
      std::string tmp = std::string(trimmed_view_(line));
    }
  }

  // finally, read the column names:
  std::vector<std::string> parsed_col_names;
  {
    std::string line;
    std::getline(stream, line);
    TokenFinder<2> f{line, 0};

    for (std::string_view tok = f.next(); not tok.empty(); tok = f.next()) {
      parsed_col_names.push_back(std::string(tok));
    }
  }

  // confirm that the column names are consistent with our expectations (for
  // the specified table-kind) and retrieve better names for the columns
  // - these improved column names reflect the fact that file is structured to
  //   have nested columns. The upper level is tricky to parse so we just parse
  //   the bottom layer. As a consequence, our parsed_col_names doesn't contain
  //   unique vals. The better names are unique
  std::vector<std::string> col_names = better_col_names(parsed_col_names, kind);
  const std::size_t ncols = col_names.size();

  // finally, read the actual data:
  std::vector<double> data = {};
  while (stream.good()) {
    std::string line;
    std::getline(stream, line);
    if (trimmed_view_(line).size() == 0) break;

    TokenFinder<2> f{line, 0};

    for (std::string_view tok = f.next(); not tok.empty(); tok = f.next()) {
      char* ptr_end{};
      errno = 0;
      // note the string of characters in tok is guaranteed to be followed
      // (eventually) by a null character since tok is a view of line.data()
      double val = std::strtod(tok.data(), &ptr_end);

      if ((errno != 0) or ((ptr_end - tok.data()) != tok.size())){
        CHOLLA_ERROR("error parsing floating-point val in %s on line: \"%s\"",
                     fname.c_str(), line.c_str());
      }
      data.push_back(val);
    }

    // sanity check:
    if ((data.size() % ncols) != 0) CHOLLA_ERROR("parsed wrong # of cols");
  }

  if (data.size() == 0) {
    CHOLLA_ERROR("No data parsed from %s. Is the table empty?", fname.c_str());
  }
  const std::size_t nrows = data.size() / ncols;

  return feedback::S99Table(col_names, data, ncols, nrows);
}


