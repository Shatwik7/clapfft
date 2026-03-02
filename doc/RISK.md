# Risk Environment: FFTW Plan Management in `clapfft::PlanCache<T>`

The current implementation of the FFT plan caching mechanism in `include/clapfft/fft_plan_cache.hpp` presents several architectural risks regarding memory management, null-pointer safety, and thread synchronization:

1. **Resource Accumulation (Memory Leaks):**
    FFTW plans stored within the `clapfft::PlanCache<T>::cache` map are only explicitly released when `clapfft::PlanCache<T>::cleanup()` is invoked. If the host application fails to call `cleanup()` before the object lifecycle ends, the allocated plan memory remains resident until the process terminates. This behavior is frequently flagged as "still reachable" or a memory leak by diagnostic tools like Valgrind.

2. **Null-Pointer Dereference in Execution APIs:**
    The `clapfft::PlanCache<T>::get_or_create` method has a failure path where it can store a `Wrapper` object containing a `nullptr` plan if the underlying FFTW plan creation fails. High-level execution APIs—specifically `clapfft::FFT::c2c_1d`, `clapfft::FFT::r2c_1d`, `clapfft::FFT::c2r_1d`, and `clapfft::FFT::r2r_1d`—do not currently implement a guard clause to check for `nullptr` before attempting to execute the plan. This can lead to segmentation faults during runtime if plan creation was unsuccessful.

3. **Thread Safety and Race Conditions during Cleanup:**
    There is a lack of synchronization between the global `clapfft::PlanCache<T>::cleanup()` method and the per-plan `Wrapper::exec_mutex`. The `cleanup()` function destroys plans immediately without acquiring the specific mutex associated with each plan. Consequently, if one thread invokes `cleanup()` while another thread is concurrently executing a transform via the same plan, a race condition occurs, leading to undefined behavior or crashes due to the use of a destroyed plan.
