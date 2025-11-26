#ifndef FPS_TIMER_H
#define FPS_TIMER_H

#include <chrono>

class FpsTimer {
public:
    // Constructor. Sets the sampling frequency.
    explicit FpsTimer(int sampleFrequency = 1);

    // Starts a new frame and increments the frame count.
    void startNewFrame();

    // Records the time when preprocessing is completed.
    void recordPreprocessEnd();
    
    // Records the time when inference is completed.
    void recordInferenceEnd();
    
    // Records the time when output is completed.
    void recordOutputEnd();
    
    // Prints the performance metrics based on the sampling frequency.
    void printResults() const;

private:
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point frameStartTime;
    std::chrono::steady_clock::time_point preprocessEndTime;
    std::chrono::steady_clock::time_point inferenceEndTime;
    std::chrono::steady_clock::time_point outputEndTime;

    long long frameCount;
    int sampleFrequency;
};

#endif // FPS_TIMER_H
