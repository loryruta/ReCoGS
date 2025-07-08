#pragma once

#include <cassert>
#include <cstdarg>
#include <exception>
#include <string>
#include <utility>

#include <fmt/format.h>

#define CHECK_STATE(condition, ...) recogs::check_state(!!(condition), #condition, __FILE__, __LINE__, ##__VA_ARGS__)

#define CHECK_ARG(condition, ...) recogs::check_arg(!!(condition), #condition, __FILE__, __LINE__, ##__VA_ARGS__)

#define DEFINE_EXCEPTION(ClassName_)                                                                                   \
    class ClassName_ : public CustomException                                                                          \
    {                                                                                                                  \
    public:                                                                                                            \
        template <typename... ARGS>                                                                                    \
        explicit ClassName_(const char* fmt_str, ARGS&&... args)                                                       \
            : CustomException(fmt::format(fmt::runtime(fmt_str), std::forward<ARGS>(args)...))                         \
        {                                                                                                              \
        }                                                                                                              \
    };

namespace recogs
{
///
class CustomException : public std::exception
{
private:
    std::string m_error;

public:
    explicit CustomException(std::string error) : m_error(std::move(error)) {}

    [[nodiscard]] const char* what() const noexcept override { return m_error.c_str(); }
};

DEFINE_EXCEPTION(IllegalArgumentException);
DEFINE_EXCEPTION(IllegalStateException);

template <typename... ARGS>
void check_state(bool condition,
                 char const* condition_str,
                 char const* file,
                 int line,
                 const char* additional_message_fmt,
                 ARGS&&... args)
{
    if (!condition) [[unlikely]] {
        std::string fmt = "Illegal state \"{}\"";
#ifndef NDEBUG
        fmt += std::string(" ({}:{})");
#endif
        if (additional_message_fmt) {
            fmt += std::string(":\n") + additional_message_fmt;
        }
#ifndef NDEBUG
        throw IllegalStateException(fmt.c_str(), condition_str, file, line, std::forward<ARGS>(args)...);
#else
        throw IllegalStateException(fmt.c_str(), condition_str, std::forward<ARGS>(args)...);
#endif
    }
}

template <typename... ARGS>
void check_arg(bool condition,
               char const* condition_str,
               char const* file,
               int line,
               const char* additional_message_fmt,
               ARGS&&... args)
{
    if (!condition) [[unlikely]] {
        std::string fmt = "Illegal argument \"{}\"";
#ifndef NDEBUG
        fmt += std::string(" ({}:{})");
#endif
        if (additional_message_fmt) {
            fmt += std::string(":\n") + additional_message_fmt;
        }
#ifndef NDEBUG
        throw IllegalArgumentException(fmt.c_str(), condition_str, file, line, std::forward<ARGS>(args)...);
#else
        throw IllegalArgumentException(fmt.c_str(), condition_str, std::forward<ARGS>(args)...);
#endif
    }
}

inline void check_state(bool condition, char const* condition_str, char const* file, int line)
{
    check_state(condition, condition_str, file, line, nullptr);
}

inline void check_arg(bool condition, char const* condition_str, char const* file, int line)
{
    check_arg(condition, condition_str, file, line, nullptr);
}
} // namespace recogs
