#include "UnicodeEncoder.hpp"

#include <vector>
#include <algorithm>

static std::unordered_map<unsigned char, std::string>
bytes_to_unicode()
{
    std::vector<int> bs;

    for (int i = '!'; i <= '~'; i++)
        bs.push_back(i);

    for (int i = 0xA1; i <= 0xAC; i++)
        bs.push_back(i);

    for (int i = 0xAE; i <= 0xFF; i++)
        bs.push_back(i);

    std::vector<int> cs = bs;

    int n = 0;

    for (int b = 0; b < 256; b++)
    {
        if (
            std::find(
                bs.begin(),
                bs.end(),
                b) == bs.end())
        {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    std::unordered_map<unsigned char, std::string> encoder;

    for (size_t i = 0; i < bs.size(); i++)
    {
        unsigned char byte = bs[i];

        int codepoint = cs[i];

        std::string utf8;

        if (codepoint <= 0x7F)
        {
            utf8.push_back(
                static_cast<char>(codepoint));
        }
        else if (codepoint <= 0x7FF)
        {
            utf8.push_back(
                0xC0 | (codepoint >> 6));

            utf8.push_back(
                0x80 | (codepoint & 0x3F));
        }
        else
        {
            utf8.push_back(
                0xE0 | (codepoint >> 12));

            utf8.push_back(
                0x80 | ((codepoint >> 6) & 0x3F));

            utf8.push_back(
                0x80 | (codepoint & 0x3F));
        }

        encoder[byte] = utf8;
    }

    return encoder;
}

UnicodeEncoder::UnicodeEncoder()
    : encoder(bytes_to_unicode())
{
}

std::string UnicodeEncoder::encode(
    const std::string &text)
{
    std::string result;

    for (unsigned char byte : text)
    {
        result += encoder.at(byte);
    }

    return result;
}