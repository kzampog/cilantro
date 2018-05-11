#pragma once

#include <chrono>

namespace cilantro {
    class Timer {
    public:
        typedef decltype(std::chrono::high_resolution_clock::now()) TimePoint;

        inline Timer() {}

        inline Timer& start() { start_time_ = std::chrono::high_resolution_clock::now(); return *this; }

        inline Timer& stop() { stop_time_ = std::chrono::high_resolution_clock::now(); return *this; }

        template <class Period = std::milli>
        inline double getElapsedTime() const {
            return std::chrono::duration<double,Period>(stop_time_ - start_time_).count();
        }

    private:
        TimePoint start_time_;
        TimePoint stop_time_;
    };
}
