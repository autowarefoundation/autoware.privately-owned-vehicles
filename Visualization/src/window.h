#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <memory>
#include "io/openlane_loader.h"
#include "visualization/annotation_visualizer.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
    // Data
    std::vector<FrameAnnotation> mAnnotations;
    std::unique_ptr<AnnotationVisualizer> mpVisualizer;
    int mCurrentFrameIndex;
    bool mIsPlaying;
    
    // UI Components
    QLabel* mpImageLabel;
    QPushButton* mpPlayPauseButton;
    QPushButton* mpLoadButton;
    QSlider* mpFrameSlider;
    QLabel* mpStatusLabel;
    QTimer* mpPlayTimer;
    
    // Methods
    void setupUI();
    void loadDataset();
    void selectFolders();
    void loadFromFolders(const std::string& jsonFolder, const std::string& imageFolder, const std::string& laneFolder = "");
    void displayFrame(int frameIndex);
    void updateStatusLabel();
    void togglePlayPause();

private slots:
    void onPlayTimerTimeout();
    void onSliderValueChanged(int value);
    void onPlayPauseClicked();
    void onLoadClicked();

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;
};