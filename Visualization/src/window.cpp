#include "window.h"

#include <iostream>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QPixmap>
#include <QImage>
#include <QFileDialog>
#include <QMessageBox>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

MainWindow::MainWindow(QWidget* parent) 
    : QMainWindow(parent), mCurrentFrameIndex(0), mIsPlaying(false)
{
    setupUI();
    mpVisualizer = std::make_unique<AnnotationVisualizer>();
    
    setWindowTitle("AutowareHMI - Select Folders to Load");
    resize(1600, 1200);  // Larger window to accommodate full-size images
}

void MainWindow::setupUI()
{
    auto centralWidget = new QWidget(this);
    auto mainLayout = new QVBoxLayout();
    
    // Create scroll area for full-size image display
    auto scrollArea = new QScrollArea();
    scrollArea->setBackgroundRole(QPalette::Dark);
    scrollArea->setStyleSheet("QScrollArea { background-color: black; }");
    
    // Image display label
    mpImageLabel = new QLabel();
    mpImageLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    mpImageLabel->setStyleSheet("QLabel { background-color: black; }");
    mpImageLabel->setScaledContents(false);  // Don't scale, keep true size
    scrollArea->setWidget(mpImageLabel);
    scrollArea->setWidgetResizable(true);  // Allow scroll area to resize
    
    mainLayout->addWidget(scrollArea);
    
    // Control panel
    auto controlLayout = new QHBoxLayout();
    
    mpLoadButton = new QPushButton("Load Dataset");
    mpPlayPauseButton = new QPushButton("Play");
    mpFrameSlider = new QSlider(Qt::Horizontal);
    mpStatusLabel = new QLabel("No dataset loaded");
    
    mpFrameSlider->setMinimum(0);
    mpFrameSlider->setMaximum(0);
    
    controlLayout->addWidget(mpLoadButton);
    controlLayout->addWidget(mpPlayPauseButton);
    controlLayout->addWidget(new QLabel("Frame:"));
    controlLayout->addWidget(mpFrameSlider);
    controlLayout->addWidget(mpStatusLabel);
    
    mainLayout->addLayout(controlLayout);
    
    centralWidget->setLayout(mainLayout);
    this->setCentralWidget(centralWidget);
    
    // Setup timer for playback
    mpPlayTimer = new QTimer(this);
    mpPlayTimer->setInterval(100);  // 10 fps
    
    // Connect signals
    QObject::connect(mpLoadButton, &QPushButton::clicked, this, &MainWindow::onLoadClicked);
    QObject::connect(mpPlayPauseButton, &QPushButton::clicked, this, &MainWindow::onPlayPauseClicked);
    QObject::connect(mpFrameSlider, &QSlider::valueChanged, this, &MainWindow::onSliderValueChanged);
    QObject::connect(mpPlayTimer, &QTimer::timeout, this, &MainWindow::onPlayTimerTimeout);
}

void MainWindow::loadDataset()
{
    selectFolders();
}

void MainWindow::selectFolders()
{
    // Select JSON annotation folder
    QString jsonFolder = QFileDialog::getExistingDirectory(
        this,
        "Select JSON Annotations Folder",
        "../autoware-hmi/dataset/cipo",
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (jsonFolder.isEmpty()) {
        mpStatusLabel->setText("No folder selected");
        return;
    }
    
    // Select image folder
    QString imageFolder = QFileDialog::getExistingDirectory(
        this,
        "Select Images Folder",
        "../autoware-hmi/dataset/images",
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (imageFolder.isEmpty()) {
        mpStatusLabel->setText("No image folder selected");
        return;
    }
    
    // Select lane annotation folder
    QString laneFolder = QFileDialog::getExistingDirectory(
        this,
        "Select Lane Annotations Folder",
        "../autoware-hmi/dataset/lane3d",
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    // Empty laneFolder is acceptable (user can cancel)
    loadFromFolders(jsonFolder.toStdString(), imageFolder.toStdString(), laneFolder.toStdString());
}

void MainWindow::loadFromFolders(const std::string& jsonFolder, const std::string& imageFolder, const std::string& laneFolder)
{
    try {
        mpStatusLabel->setText("Loading...");
        
        OpenLaneLoader loader;
        mAnnotations = loader.loadFromFolder(jsonFolder, imageFolder, laneFolder);
        
        if (!mAnnotations.empty()) {
            mpFrameSlider->setMaximum(mAnnotations.size() - 1);
            mCurrentFrameIndex = 0;
            displayFrame(0);
            updateStatusLabel();
            setWindowTitle(QString("AutowareHMI - %1 frames loaded").arg(mAnnotations.size()));
        } else {
            mpStatusLabel->setText("No annotations found in selected folders");
            QMessageBox::warning(this, "No Data", "No valid annotations found in the selected folders.");
        }
    } catch (const std::exception& e) {
        mpStatusLabel->setText(QString("Error loading dataset: ") + e.what());
        QMessageBox::critical(this, "Error", QString("Failed to load dataset:\n") + e.what());
    }
}

void MainWindow::displayFrame(int frameIndex)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(mAnnotations.size())) {
        return;
    }
    
    mCurrentFrameIndex = frameIndex;
    mpFrameSlider->blockSignals(true);
    mpFrameSlider->setValue(frameIndex);
    mpFrameSlider->blockSignals(false);
    
    const auto& annotation = mAnnotations[frameIndex];
    
    // Use new-format imagePath if available (absolute path made by loader), otherwise fall back
    std::string fullPath;
    if (!annotation.imagePath.empty()) {
        fullPath = annotation.imagePath;
    } else {
        mpStatusLabel->setText("No image found for this frame");
        return;
    }
    
    // Load image with OpenCV
    cv::Mat image = cv::imread(fullPath);
    if (image.empty()) {
        // Image not found - create a blank placeholder
        mpStatusLabel->setText(QString("Image not found (showing annotation metadata only): ") + fullPath.c_str());
        image = cv::Mat(600, 800, CV_8UC3, cv::Scalar(40, 40, 40)); // Dark gray placeholder
        
        // Draw text indicating missing image
        cv::putText(image, "Image file not found", cv::Point(200, 280), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(200, 200, 200), 2);
        cv::putText(image, fullPath, cv::Point(50, 320), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);
    }
    
    // Render annotations
    cv::Mat annotated = mpVisualizer->renderAnnotations(image, annotation);
    
    // Convert BGR to RGB for Qt
    cv::Mat rgbImage;
    cv::cvtColor(annotated, rgbImage, cv::COLOR_BGR2RGB);
    
    // Convert to QPixmap - make a deep copy to ensure data persists
    QImage qImage(rgbImage.data, rgbImage.cols, rgbImage.rows, 
                  static_cast<int>(rgbImage.step), QImage::Format_RGB888);
    // Create a copy to ensure the data persists beyond this function
    QPixmap pixmap = QPixmap::fromImage(qImage.copy());
    
    mpImageLabel->setPixmap(pixmap);
    updateStatusLabel();
}

void MainWindow::updateStatusLabel()
{
    QString status = QString("Frame %1 / %2").arg(mCurrentFrameIndex + 1).arg(mAnnotations.size());
    mpStatusLabel->setText(status);
}

void MainWindow::togglePlayPause()
{
    mIsPlaying = !mIsPlaying;
    mpPlayPauseButton->setText(mIsPlaying ? "Pause" : "Play");
    
    if (mIsPlaying) {
        mpPlayTimer->start();
    } else {
        mpPlayTimer->stop();
    }
}

void MainWindow::onPlayTimerTimeout()
{
    if (mCurrentFrameIndex < static_cast<int>(mAnnotations.size()) - 1) {
        displayFrame(mCurrentFrameIndex + 1);
    } else {
        // Loop back to start
        displayFrame(0);
    }
}

void MainWindow::onSliderValueChanged(int value)
{
    if (!mIsPlaying) {
        displayFrame(value);
    }
}

void MainWindow::onPlayPauseClicked()
{
    togglePlayPause();
}

void MainWindow::onLoadClicked()
{
    loadDataset();
}