#include "fps_timer.hpp"
#include "logging.hpp"

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
        LOG_WARN("No frames processed yet.");
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

    LOG_INFO("--------------------------\n");
    LOG_INFO("--- Performance Metrics ---\n");
    LOG_INFO("* Total frames processed: %lld\n", frameCount);
    LOG_INFO("* Current FPS: %.2f\n", fps);
    LOG_INFO("--- Per-frame Timing (microseconds) ---\n");
    LOG_INFO("* Total processing time: %ld us\n", totalFrameDuration);
    LOG_INFO("* Preprocessing time: %ld us\n", preprocessDuration);
    LOG_INFO("* Inference time: %ld us\n", inferenceDuration);
    LOG_INFO("* Output time: %ld us\n", outputDuration);
    LOG_INFO("--------------------------\n");
    fflush(stdout);
}
