#include <iostream>
#include <iomanip>

#include "fps_timer.hpp"

// Constructor that sets the sampling frequency.
FpsTimer::FpsTimer(int sampleFrequency) : frameCount(0), sampleFrequency(sampleFrequency) {
    startTime = std::chrono::steady_clock::now();
}

// Starts a new frame and increments the frame count.
void FpsTimer::startNewFrame() {
    frameStartTime = std::chrono::steady_clock::now();
    frameCount++;
}

// Records the time when preprocessing is completed.
void FpsTimer::recordPreprocessEnd() {
    preprocessEndTime = std::chrono::steady_clock::now();
}


// Records the time when inference is completed.
void FpsTimer::recordInferenceEnd() {
    inferenceEndTime = std::chrono::steady_clock::now();
}

// Records the time when output is completed.
void FpsTimer::recordOutputEnd() {
    outputEndTime = std::chrono::steady_clock::now();
    
    // Check if the current frame is a sampling point
    if (frameCount != 0 && frameCount % sampleFrequency == 0) {
        printResults();
    }
}

// Prints the performance metrics.
void FpsTimer::printResults() const {
    if (frameCount == 0) {
        std::cout << "No frames processed yet." << std::endl;
        return;
    }

    // Calculate per-frame durations in microseconds
    auto totalFrameDuration = std::chrono::duration_cast<std::chrono::microseconds>(outputEndTime - frameStartTime).count();
    auto preprocessDuration = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEndTime - frameStartTime).count(); // New duration
    auto inferenceDuration = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEndTime - preprocessEndTime).count(); // Adjusted calculation
    auto outputDuration = std::chrono::duration_cast<std::chrono::microseconds>(outputEndTime - inferenceEndTime).count();
    
    // Calculate overall FPS
    auto overallDuration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count();
    double fps = (overallDuration > 0) ? (static_cast<double>(frameCount) / overallDuration) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "--- Performance Metrics ---" << std::endl;
    std::cout << "* Total frames processed: " << frameCount << std::endl;
    std::cout << "* Current FPS: " << fps << std::endl;
    std::cout << "--- Per-frame Timing (microseconds) ---" << std::endl;
    std::cout << "* Total processing time: " << totalFrameDuration << " µs" << std::endl;
    std::cout << "* Preprocessing time: " << preprocessDuration << " µs" << std::endl; // New output
    std::cout << "* Inference time: " << inferenceDuration << " µs" << std::endl;
    std::cout << "* Output time: " << outputDuration << " µs" << std::endl;
    std::cout << "--------------------------" << std::endl;
}
