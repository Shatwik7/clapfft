#ifndef CLAPFFT_WISDOM_HPP
#define CLAPFFT_WISDOM_HPP

#include <string>

namespace clapfft
{
    struct Wisdom
    {
        template <typename T>
        static bool import_from_filename(const std::string &filename);

        template <typename T>
        static void export_to_filename(const std::string &filename);

        template <typename T>
        static std::string export_to_string();

        template <typename T>
        static bool import_from_string(const std::string &input_string);
    };
} // namespace clapfft

#endif // CLAPFFT_WISDOM_HPP