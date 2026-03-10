#pragma once
#include <clapfft/fft_traits.hpp>
#include "fft_flags.hpp" // planning flag definitions
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstddef>
#include <vector>
#include <complex>

namespace clapfft
{

    enum R2RKind
    {
        FFT_R2HC = 0,
        FFT_HC2R = 1,
        FFT_DHT = 2,
        FFT_REDFT00 = 3,
        FFT_REDFT01 = 4,
        FFT_REDFT10 = 5,
        FFT_REDFT11 = 6,
        FFT_RODFT00 = 7,
        FFT_RODFT01 = 8,
        FFT_RODFT10 = 9,
        FFT_RODFT11 = 10
    };
    enum class TransformKind
    {
        C2C,
        C2R,
        R2C,
        R2R
    };

    struct PlanKey
    {
        TransformKind kind;
        int dim;
        int n0, n1, n2;
        int sign;
        int k0, k1, k2;
        fft_flags flags; // planning options (measure/estimate/etc.)

        bool operator==(const PlanKey &o) const
        {
            return kind == o.kind && dim == o.dim && n0 == o.n0 && n1 == o.n1 &&
                   n2 == o.n2 && sign == o.sign && k0 == o.k0 && k1 == o.k1 && k2 == o.k2 &&
                   flags == o.flags;
        }
    };

    struct PlanKeyHash
    {
        size_t operator()(const PlanKey &k) const
        {
            size_t h = std::hash<int>()(static_cast<int>(k.kind));
            h ^= std::hash<int>()(k.dim) << 1;
            h ^= std::hash<int>()(k.n0) << 2;
            h ^= std::hash<int>()(k.n1) << 3;
            h ^= std::hash<int>()(k.n2) << 4;
            h ^= std::hash<int>()(k.sign) << 5;
            h ^= std::hash<int>()(k.k0) << 6;
            h ^= std::hash<int>()(k.k1) << 7;
            h ^= std::hash<int>()(k.k2) << 8;
            h ^= std::hash<fft_flags>()(k.flags) << 9;
            return h;
        }
    };

    template <typename T>
    class PlanCache
    {
    public:
        using traits = fft_trait<T>;
        using plan_type = typename traits::plan_type;

        struct Wrapper
        {
            plan_type plan;
            std::mutex exec_mutex;
        };

    private:
        template <typename Factory>
        static std::shared_ptr<Wrapper> get_or_create(const PlanKey &key, Factory &&factory)
        {
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                auto it = cache.find(key);
                if (it != cache.end())
                {
                    return it->second;
                }
            }

            auto wrapper = std::make_shared<Wrapper>();
            {
                std::lock_guard<std::mutex> planner_lock(planner_mutex);
                wrapper->plan = factory();
            }

            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                std::pair<typename std::unordered_map<PlanKey, std::shared_ptr<Wrapper>, PlanKeyHash>::iterator, bool> result = cache.emplace(key, wrapper);
                auto it = result.first;
                bool inserted = result.second;
                if (!inserted)
                {
                    traits::destroy_plan(wrapper->plan);
                    return it->second;
                }
                return wrapper;
            }
        }

        static std::size_t element_count(int dim, int n0, int n1, int n2)
        {
            if (dim == 1)
            {
                return static_cast<std::size_t>(n0);
            }
            if (dim == 2)
            {
                return static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1);
            }
            return static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
        }

        static std::unordered_map<
            PlanKey,
            std::shared_ptr<Wrapper>,
            PlanKeyHash>
            cache;

        static std::mutex cache_mutex;
        static std::mutex planner_mutex;

    public:
        static std::shared_ptr<Wrapper> get_c2c_1d(int n, int sign,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2C, 1, n, 1, 1, sign, 0, 0, 0, flags};
            return get_or_create(key, [n, sign, flags]()
                                 {
            std::vector<std::complex<T>> dummy_in(static_cast<std::size_t>(n));
            std::vector<std::complex<T>> dummy_out(static_cast<std::size_t>(n));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_in.data());
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_out.data());
            return traits::plan_dft_1d(n, in_ptr, out_ptr, sign, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_c2c_2d(int n0, int n1, int sign,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2C, 2, n0, n1, 1, sign, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, sign, flags]()
                                 {
            std::vector<std::complex<T>> dummy_in(element_count(2, n0, n1, 1));
            std::vector<std::complex<T>> dummy_out(element_count(2, n0, n1, 1));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_in.data());
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_out.data());
            return traits::plan_dft_2d(n0, n1, in_ptr, out_ptr, sign, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_c2c_3d(int n0, int n1, int n2, int sign,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2C, 3, n0, n1, n2, sign, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, n2, sign, flags]()
                                 {
            std::vector<std::complex<T>> dummy_in(element_count(3, n0, n1, n2));
            std::vector<std::complex<T>> dummy_out(element_count(3, n0, n1, n2));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_in.data());
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(dummy_out.data());
            return traits::plan_dft_3d(n0, n1, n2, in_ptr, out_ptr, sign, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2c_1d(int n,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2C, 1, n, 1, 1, 0, 0, 0, 0, flags};
            return get_or_create(key, [n, flags]()
                                 {
            std::vector<T> real_dummy(static_cast<std::size_t>(n));
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n / 2 + 1));
            auto in_ptr = real_dummy.data();
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            return traits::plan_dft_r2c_1d(n, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2c_2d(int n0, int n1,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2C, 2, n0, n1, 1, 0, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, flags]()
                                 {
            std::vector<T> real_dummy(element_count(2, n0, n1, 1));
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1 / 2 + 1));
            auto in_ptr = real_dummy.data();
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            return traits::plan_dft_r2c_2d(n0, n1, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2c_3d(int n0, int n1, int n2,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2C, 3, n0, n1, n2, 0, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, n2, flags]()
                                 {
            std::vector<T> real_dummy(element_count(3, n0, n1, n2));
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2 / 2 + 1));
            auto in_ptr = real_dummy.data();
            auto out_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            return traits::plan_dft_r2c_3d(n0, n1, n2, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_c2r_1d(int n,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2R, 1, n, 1, 1, 0, 0, 0, 0, flags};
            return get_or_create(key, [n, flags]()
                                 {
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n / 2 + 1));
            std::vector<T> real_dummy(static_cast<std::size_t>(n));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            auto out_ptr = real_dummy.data();
            return traits::plan_dft_c2r_1d(n, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_c2r_2d(int n0, int n1,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2R, 2, n0, n1, 1, 0, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, flags]()
                                 {
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1 / 2 + 1));
            std::vector<T> real_dummy(element_count(2, n0, n1, 1));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            auto out_ptr = real_dummy.data();
            return traits::plan_dft_c2r_2d(n0, n1, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_c2r_3d(int n0, int n1, int n2,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::C2R, 3, n0, n1, n2, 0, 0, 0, 0, flags};
            return get_or_create(key, [n0, n1, n2, flags]()
                                 {
            std::vector<std::complex<T>> complex_dummy(static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2 / 2 + 1));
            std::vector<T> real_dummy(element_count(3, n0, n1, n2));
            auto in_ptr = reinterpret_cast<typename traits::complex_type*>(complex_dummy.data());
            auto out_ptr = real_dummy.data();
            return traits::plan_dft_c2r_3d(n0, n1, n2, in_ptr, out_ptr, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2r_1d(int n, fftw_r2r_kind kind,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2R, 1, n, 1, 1, 0, static_cast<int>(kind), 0, 0, flags};
            return get_or_create(key, [n, kind, flags]()
                                 {
            std::vector<T> real_dummy_in(static_cast<std::size_t>(n));
            std::vector<T> real_dummy_out(static_cast<std::size_t>(n));
            return traits::plan_r2r_1d(n, real_dummy_in.data(), real_dummy_out.data(), kind, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2r_2d(int n0, int n1, fftw_r2r_kind kind0, fftw_r2r_kind kind1,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2R, 2, n0, n1, 1, 0, static_cast<int>(kind0), static_cast<int>(kind1), 0, flags};
            return get_or_create(key, [n0, n1, kind0, kind1, flags]()
                                 {
            std::vector<T> real_dummy_in(element_count(2, n0, n1, 1));
            std::vector<T> real_dummy_out(element_count(2, n0, n1, 1));
            return traits::plan_r2r_2d(n0, n1, real_dummy_in.data(), real_dummy_out.data(), kind0, kind1, flags | CLAP_FFT_UNALIGNED); });
        }

        static std::shared_ptr<Wrapper> get_r2r_3d(int n0, int n1, int n2, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2,
                                                   fft_flags flags = CLAP_FFT_ESTIMATE)
        {
            PlanKey key{TransformKind::R2R, 3, n0, n1, n2, 0, static_cast<int>(kind0), static_cast<int>(kind1), static_cast<int>(kind2), flags};
            return get_or_create(key, [n0, n1, n2, kind0, kind1, kind2, flags]()
                                 {
            std::vector<T> real_dummy_in(element_count(3, n0, n1, n2));
            std::vector<T> real_dummy_out(element_count(3, n0, n1, n2));
            return traits::plan_r2r_3d(n0, n1, n2, real_dummy_in.data(), real_dummy_out.data(), kind0, kind1, kind2, flags | CLAP_FFT_UNALIGNED); });
        }

        static void cleanup()
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            for (typename std::unordered_map<PlanKey, std::shared_ptr<Wrapper>, PlanKeyHash>::iterator it = cache.begin(); it != cache.end(); ++it)
            {
                traits::destroy_plan(it->second->plan);
            }
            cache.clear();
        }
    };

    template <typename T>
    std::unordered_map<
        PlanKey,
        std::shared_ptr<typename PlanCache<T>::Wrapper>,
        PlanKeyHash>
        PlanCache<T>::cache;

    template <typename T>
    std::mutex PlanCache<T>::cache_mutex;

    template <typename T>
    std::mutex PlanCache<T>::planner_mutex;

}