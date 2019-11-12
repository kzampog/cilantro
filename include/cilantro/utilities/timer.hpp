#pragma once

#include <chrono>

namespace cilantro {
    class Timer {
    public:
        inline Timer(bool start = false) { if (start) start_time_ = std::chrono::high_resolution_clock::now(); }

        inline Timer& start() { start_time_ = std::chrono::high_resolution_clock::now(); return *this; }

        inline Timer& stop() { stop_time_ = std::chrono::high_resolution_clock::now(); return *this; }

        template <class DurationT = std::chrono::duration<double,std::milli>>
        inline typename DurationT::rep getElapsedTimeSinceStart() const {
            return std::chrono::duration_cast<DurationT>(std::chrono::high_resolution_clock::now() - start_time_).count();
        }

        template <class DurationT = std::chrono::duration<double,std::milli>>
        inline typename DurationT::rep getElapsedTime() const {
            return std::chrono::duration_cast<DurationT>(stop_time_ - start_time_).count();
        }

        template <class DurationT = std::chrono::duration<double,std::milli>>
        inline typename DurationT::rep stopAndGetElapsedTime() {
            return stop().template getElapsedTime<DurationT>();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
    };
}
