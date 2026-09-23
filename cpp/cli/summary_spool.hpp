#ifndef HIKOBOSHI_CLI_SUMMARY_SPOOL_HPP
#define HIKOBOSHI_CLI_SUMMARY_SPOOL_HPP

#include <hikoboshi/universal/status.hpp>
#include <cstdio>
#include <memory>
#include <ostream>
#include <sstream>

namespace hikoboshi::cli {

// Anonymous TSV storage, bounded in memory and replayable to multiple outputs.
// Error strings must have static lifetime, as required by universal::Status.
class SummarySpool {
 public:
  struct Errors { const char* write; const char* rewind; const char* publish; };
  explicit SummarySpool(Errors errors)
      : errors_(errors), file_(std::tmpfile(), &std::fclose) {}
  bool valid() const { return file_ != nullptr; }
  universal::Status append(std::ostringstream& rows) {
    const auto text = rows.str();
    if (!file_ || std::fwrite(text.data(), 1, text.size(), file_.get()) != text.size())
      return {universal::StatusCode::Unavailable, errors_.write};
    dirty_ = true;
    rows.str({});
    return universal::ok_status();
  }
  universal::Status publish(std::ostream& output) {
    if (!file_ || (dirty_ && std::fflush(file_.get())) ||
        std::fseek(file_.get(), 0, SEEK_SET))
      return {universal::StatusCode::Unavailable, errors_.rewind};
    dirty_ = false;
    char buffer[65536];
    for (;;) {
      const auto count = std::fread(buffer, 1, sizeof(buffer), file_.get());
      output.write(buffer, static_cast<std::streamsize>(count));
      if (!output || std::ferror(file_.get()))
        return {universal::StatusCode::Unavailable, errors_.publish};
      if (count < sizeof(buffer)) return universal::ok_status();
    }
  }
 private:
  Errors errors_;
  std::unique_ptr<std::FILE, int(*)(std::FILE*)> file_;
  bool dirty_ = false;
};

}  // namespace hikoboshi::cli
#endif
