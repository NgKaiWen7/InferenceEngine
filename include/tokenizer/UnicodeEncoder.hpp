#pragma once

#include <string>
#include <unordered_map>


class UnicodeEncoder
{
public:

    UnicodeEncoder();

    std::string encode(
        const std::string& text
    );


private:

    std::unordered_map<unsigned char, std::string> encoder;
};