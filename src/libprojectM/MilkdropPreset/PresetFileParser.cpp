#include "PresetFileParser.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace libprojectM {
namespace MilkdropPreset {

auto PresetFileParser::Read(const std::string& presetFile) -> bool
{
    std::ifstream presetStream(presetFile.c_str(), std::ios_base::in | std::ios_base::binary);
    return Read(presetStream);
}

auto PresetFileParser::Read(std::istream& presetStream) -> bool
{
    if (!presetStream.good())
    {
        return false;
    }

    presetStream.seekg(0, presetStream.end);
    auto fileSize = presetStream.tellg();
    presetStream.seekg(0, presetStream.beg);

    if (static_cast<size_t>(fileSize) > maxFileSize)
    {
        return false;
    }

    std::vector<char> presetFileContents(fileSize);
    presetStream.read(presetFileContents.data(), fileSize);

    if (presetStream.fail() || presetStream.bad())
    {
        return false;
    }

    std::string_view fileView(presetFileContents.data(), presetFileContents.size());
    size_t startPos{0};
    size_t endPos = presetFileContents.size();

    // In MilkDrop 3.2 Double-Presets, Preset 2 is the primary preset matching the title/filename,
    // while Preset 1 is a background/ambient layer. Load Preset 2 when present.
    size_t p2Pos = fileView.find("[PRESET2_BEGIN]");
    if (p2Pos != std::string_view::npos)
    {
        startPos = p2Pos;
        size_t p2End = fileView.find("[PRESET2_END]", p2Pos);
        if (p2End != std::string_view::npos)
        {
            endPos = p2End;
        }
    }
    else
    {
        size_t p1Pos = fileView.find("[PRESET1_BEGIN]");
        if (p1Pos != std::string_view::npos)
        {
            startPos = p1Pos;
            size_t p1End = fileView.find("[PRESET1_END]", p1Pos);
            if (p1End != std::string_view::npos)
            {
                endPos = p1End;
            }
        }
    }

    size_t lineStart = startPos;
    size_t pos = startPos;

    auto parseLineIfDataAvailable = [this, &pos, &lineStart, &presetFileContents]() {
        if (pos > lineStart)
        {
            auto beg = presetFileContents.begin();
            std::string line(beg + lineStart, beg + pos);
            ParseLine(line);
        }
    };

    while (pos < endPos)
    {
        switch (presetFileContents[pos])
        {
            case '\r':
            case '\n':
                // EOL, skip over CRLF
                parseLineIfDataAvailable();
                lineStart = pos + 1;
                break;

            case '\0':
                // Null char is not expected. Could be a random binary file.
                return false;
        }

        ++pos;
    }

    parseLineIfDataAvailable();

    return !m_presetValues.empty();
}

auto PresetFileParser::GetCode(const std::string& keyPrefix) const -> std::string
{
    auto lowerKey = ToLower(keyPrefix);

    std::stringstream code;                        //!< The parsed code
    std::string key(lowerKey.length() + 5, '\0'); //!< Allocate a string that can hold up to 5 digits.

    key.replace(0, lowerKey.length(), lowerKey);

    for (int index{1}; index <= 99999; ++index)
    {
        key.replace(lowerKey.length(), 5, std::to_string(index));
        if (m_presetValues.find(key) == m_presetValues.end())
        {
            break;
        }

        auto line = m_presetValues.at(key);

        // Remove backtick char in shader code
        if (!line.empty() && line.at(0) == '`')
        {
            line.erase(0, 1);
        }
        code << line << std::endl;
    }

    auto codeStr = code.str();

    return codeStr;
}

auto PresetFileParser::GetInt(const std::string& key, int defaultValue) -> int
{
    auto lowerKey = ToLower(key);
    if (m_presetValues.find(lowerKey) != m_presetValues.end())
    {
        try
        {
            return std::stoi(m_presetValues.at(lowerKey));
        }
        catch (std::logic_error&)
        {
        }
    }

    return defaultValue;
}

auto PresetFileParser::GetFloat(const std::string& key, float defaultValue) -> float
{
    auto lowerKey = ToLower(key);
    if (m_presetValues.find(lowerKey) != m_presetValues.end())
    {
        try
        {
            return std::stof(m_presetValues.at(lowerKey));
        }
        catch (std::logic_error&)
        {
        }
    }

    return defaultValue;
}

auto PresetFileParser::GetBool(const std::string& key, bool defaultValue) -> bool
{
    return GetInt(key, static_cast<int>(defaultValue)) > 0;
}

auto PresetFileParser::GetString(const std::string& key, const std::string& defaultValue) -> std::string
{
    auto lowerKey = ToLower(key);
    if (m_presetValues.find(lowerKey) != m_presetValues.end())
    {
        return m_presetValues.at(lowerKey);
    }

    return defaultValue;
}

const std::map<std::string, std::string>& PresetFileParser::PresetValues() const
{
    return m_presetValues;
}

void PresetFileParser::ParseLine(const std::string& line)
{
    // Search for first delimiter, either space or equal
    auto varNameDelimiterPos = line.find_first_of(" =");

    if (varNameDelimiterPos == std::string::npos || varNameDelimiterPos == 0)
    {
        // Empty line, delimiter at start of line or no delimiter found, skip.
        return;
    }

    // Convert key to lower case, as INI functions are not case-sensitive.
    std::string varName(ToLower(std::string(line.begin(), line.begin() + varNameDelimiterPos)));
    std::string value(line.begin() + varNameDelimiterPos + 1, line.end());

    // Only add first occurrence to mimic Milkdrop behaviour
    if (!varName.empty() && m_presetValues.find(varName) == m_presetValues.end())
    {
        m_presetValues.emplace(std::move(varName), std::move(value));
    }
}

auto PresetFileParser::ToLower(std::string str) -> std::string
{
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); }
    );

    return str;
}

} // namespace MilkdropPreset
} // namespace libprojectM
