/*! \file
 *  \brief Declare interface for generic attribute recorder interface
 *
 *  The main purpose of this file is declare @ref AttrRecorderInterface.
 *
 *  While we are at it, we also declare @ref TextAttrRecorder, the wrapper around
 *  ``std::FILE`` that implements the @ref AttrRecorderInterface interface.
 *
 *  We explicitly choose NOT to implement the equivalent for hdf5 files; that wrapper
 *  class is instead implemented in close proximity to other HDF5 logic.
 */

#include <cstdio>  // std::FILE

/*! Abstract class that provides the interface for recording file header information
 *
 *  The expectation is that logic for serializing Attribute information to different
 *  formats will be organized into different subclasses. For example, we might define
 *  an HDF5 class and a Text class.
 *
 *  \note
 *  For less experienced C++ devs:
 *  - virtual methods can be overridden by subclasses
 *  - a pure virtual method doesn't provide a default implementation. If you try to
 *    initialize a subclass that doesn't override the methods, the program won't compile
 *  - virtual methods with default implementations don't need to be overriden (in the
 *    context of this class, they may be implemented in terms of other virtual methods)
 */
class AttrRecorderInterface
{
 public:
  // I don't anticipate that we'll need the virtual destructor, but we're
  // including it just in case. I'm concerned that if we don't declare it to
  // be virtual a future change could make it necessary and people may be
  // confused about failure since inheritance is EXTREMELY rare in this
  // codebase
  virtual ~AttrRecorderInterface() {}

  // declare the recording methods for arrays of values
  ///@{
  /*! record a 1d array of values */
  virtual void record_arr(const char* name, const double* arr, int length) = 0;
  virtual void record_arr(const char* name, const int* arr, int length)    = 0;
  virtual void record_arr(const char* name, const long* arr, int length)   = 0;
  ///@}

  // the following member group consists of member functions for recording scalar
  // attributes. The overload for writing a string is the only pure virtual member.
  // Every other member function has a default implementation that treats the scalar
  // as a 1 element array

  ///@{
  /*! record a scalar attribute */
  virtual void record(const char* name, const char* val) = 0;
  virtual void record(const char* name, double val)      = 0;
  virtual void record(const char* name, int val)         = 0;
  virtual void record(const char* name, long val)        = 0;
  ///@}

  /*! A convenience method for recording 3 doubles */
  void record_triple(const char* name, double a, double b, double c)
  {
    double tmp[3] = {a, b, c};
    this->record_arr(name, tmp, 3);
  }

  /* A convenience method for recording 3 ints */
  void record_triple(const char* name, int a, int b, int c)
  {
    int tmp[3] = {a, b, c};
    this->record_arr(name, tmp, 3);
  }
};

/*! Provides a nice wrapper around an text file pointer for the purpose of recording
 *  attributes
 *
 *  We intentionally target TOML Formatting so that users can use a toml parser (like
 *  python's built-in tomllib module) to read in the header values. Users would do that
 *  that for all contents between `BEGIN-HEADER` and `END-HEADER`.
 */
class TextAttrRecorder : public AttrRecorderInterface
{
  std::FILE* fp_;

  // checks if a string needs escaping
  // this is quick and dirty
  bool key_needs_escaping_(const char* s) const
  {
    int i = 0;
    while (true) {
      if (s[i] == '\0') return false;
      if ((s[i] == ' ') or (s[i] == '\t')) return true;
      i++;
    }
  }

  template <typename T>
  void write_val_(T val)
  {
    if constexpr (std::is_same_v<T, long>) {
      std::fprintf(this->fp_, "%ld", val);
    } else if constexpr (std::is_same_v<T, int>) {
      std::fprintf(this->fp_, "%d", val);
    } else if constexpr (std::is_same_v<T, double>) {
      std::fprintf(this->fp_, "%.17g", val);
    } else if constexpr (std::is_same_v<T, const char*>) {
      if (val[0] == '\0') {
        std::fputs("\"\"", this->fp_);
      } else {
        std::fprintf(this->fp_, "'''%s'''", val);
      }
    } else {
      static_assert(always_false<T>, "unexpected type");
    }
  }

  template <typename T>
  void record_(const char* name, const T* val, int length)
  {
    if (this->key_needs_escaping_(name)) {
      // this isn't fully escaped (but good enough to start)
      std::fprintf(this->fp_, "'%s'", name);
    } else {
      std::fprintf(this->fp_, "%s", name);
    }
    std::fputs(" = ", this->fp_);
    if (length < 0) {  // denotes a scalar
      this->write_val_<T>(*val);
    } else {  // recording an array
      std::fputc('[', this->fp_);
      for (int i = 0; i < length; i++) {
        if (i > 0) {
          std::fputc(',', this->fp_);
          std::fputc(' ', this->fp_);
        }
        write_val_(val[i]);
      }
      std::fputc(']', this->fp_);
    }
    std::fputc('\n', this->fp_);
  }

 public:
  TextAttrRecorder() = delete;

  explicit TextAttrRecorder(std::FILE* fp) : fp_{fp}
  {
    CHOLLA_ASSERT(fp != nullptr, "received nullptr");
    std::fputs("BEGIN-HEADER\n", this->fp_);
  }

  // the destructor does NOT call `std::fclose(this->fp_)` since instances of this type
  // are intended to be temporary wrappers
  ~TextAttrRecorder() override { std::fputs("END-HEADER\n", this->fp_); }

  void record_arr(const char* name, const double* arr, int length) override { this->record_(name, arr, length); }

  void record_arr(const char* name, const int* arr, int length) override { this->record_(name, arr, length); }

  void record_arr(const char* name, const long* arr, int length) override { this->record_(name, arr, length); }

  void record(const char* name, const char* val) override { this->record_(name, &val, -1); }
  void record(const char* name, double val) override { this->record_(name, &val, -1); }
  void record(const char* name, int val) override { this->record_(name, &val, -1); }
  void record(const char* name, long val) override { this->record_(name, &val, -1); }
};
