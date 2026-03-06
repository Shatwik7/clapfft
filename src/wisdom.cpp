#include <clapfft/wisdom.hpp>
#include <clapfft/fft_traits.hpp>
#include <cstdlib>

namespace clapfft
{
    template <typename T>
    bool Wisdom::import_from_filename(const std::string &filename)
    {
        using traits = fft_trait<T>;
        return traits::import_wisdom_from_filename(filename.c_str()) != 0;
    }

    template <typename T>
    void Wisdom::export_to_filename(const std::string &filename)
    {
        using traits = fft_trait<T>;
        traits::export_wisdom_to_filename(filename.c_str());
    }

    template <typename T>
    std::string Wisdom::export_to_string()
    {
        using traits = fft_trait<T>;
        char *s = traits::export_wisdom_to_string();
        if (s)
        {
            std::string result(s);
            free(s); // FFTW allocates string with malloc
            return result;
        }
        return {};
    }

    template <typename T>
    bool Wisdom::import_from_string(const std::string &input_string)
    {
        using traits = fft_trait<T>;
        return traits::import_wisdom_from_string(input_string.c_str()) != 0;
    }

    // Explicit instantiations
    template bool Wisdom::import_from_filename<float>(const std::string &);
    template bool Wisdom::import_from_filename<double>(const std::string &);
    template bool Wisdom::import_from_filename<long double>(const std::string &);

    template void Wisdom::export_to_filename<float>(const std::string &);
    template void Wisdom::export_to_filename<double>(const std::string &);
    template void Wisdom::export_to_filename<long double>(const std::string &);

    template std::string Wisdom::export_to_string<float>();
    template std::string Wisdom::export_to_string<double>();
    template std::string Wisdom::export_to_string<long double>();

    template bool Wisdom::import_from_string<float>(const std::string &);
    template bool Wisdom::import_from_string<double>(const std::string &);
    template bool Wisdom::import_from_string<long double>(const std::string &);

} // namespace clapfft