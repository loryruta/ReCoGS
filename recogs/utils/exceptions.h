#pragma once

#include <exception>

namespace recogs
{
///
class CustomException : public std::exception
{
private:
    char const* const m_message;

public:
    explicit CustomException(const char* message) : m_message(message) {}

    const char* what() { return m_message; }
};

///
class IllegalArgumentException : public CustomException
{
public:
    explicit IllegalArgumentException(const char* message) : CustomException(message) {}
};

///
class IllegalStateException : public CustomException
{
public:
    explicit IllegalStateException(const char* message) : CustomException(message) {}
};
} // namespace gslab
