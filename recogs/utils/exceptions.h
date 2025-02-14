#pragma once

#include <exception>

namespace gs_train
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
} // namespace gslab
