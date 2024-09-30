#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <thread>
#include <iostream>
#include <fstream>
#include <numeric> // For std::accumulate

using namespace cv;
using namespace std;

struct TrackedObject {
    Rect bbox;
    Ptr<Tracker> tracker;
};

vector<Rect> selectBboxes(const Mat& frame) {
    vector<Rect> bboxes;
    bool select = true;
    Mat displayFrame;

    // Create a smaller frame for ROI selection
    Size reducedSize(static_cast<int>(frame.cols * 0.25), static_cast<int>(frame.rows * 0.25));
    resize(frame, displayFrame, reducedSize);

    while (select) {
        Rect bbox = selectROI("Select object (Press 'c' to cancel)", displayFrame);
        if (bbox.width > 0 && bbox.height > 0) {
            // Scale bbox coordinates back to original size
            bbox.x = static_cast<int>(bbox.x / 0.25);
            bbox.y = static_cast<int>(bbox.y / 0.25);
            bbox.width = static_cast<int>(bbox.width / 0.25);
            bbox.height = static_cast<int>(bbox.height / 0.25);
            bboxes.push_back(bbox);
        } else {
            select = false;
        }
    }

    return bboxes;
}

void drawBoundingBoxes(Mat& frame, const vector<Rect>& bboxes) {
    for (const Rect& bbox : bboxes) {
        rectangle(frame, bbox, Scalar(255, 0, 0), 2);
    }
}

int main() {
    VideoCapture cap("P1000214.MOV"); // Replace with your video path
    if (!cap.isOpened()) {
        cerr << "Error opening video capture\n";
        return 1;
    }

    Mat frame;
    if (!cap.read(frame)) {
        cerr << "Error reading first frame\n";
        return 1;
    }

    // Initialize tracker parameters
    cv::TrackerCSRT::Params params;
    params.use_gray = true;
    params.use_hog = false;
    params.use_segmentation = false;

    vector<TrackedObject> trackedObjects;
    vector<Rect> initialBboxes = selectBboxes(frame);

    for (const Rect& bbox : initialBboxes) {
        TrackedObject obj;
        obj.bbox = bbox;
        obj.tracker = TrackerCSRT::create(params); // Create tracker with specified parameters
        obj.tracker->init(frame, bbox);
        trackedObjects.push_back(obj);
    }

    vector<Mat> frames; // Vector to store all frames
    int totalFrames = 0;
    double startTime = getTickCount();
    vector<double> fpsValues; // Vector to store FPS values

    // Open CSV file for writing FPS data
    ofstream csvFile("data_thread.csv", ios::app);
    if (!csvFile.is_open()) {
        cerr << "Error opening CSV file\n";
        return 1;
    }

    // Write CSV header if file is empty
    csvFile << "FPS\n";

    const int displayWidth = int(frame.cols * 0.3);
    const int displayHeight = int(frame.rows * 0.3);

    while (cap.read(frame)) {
        Mat reducedFrame, displayFrame;
        resize(frame, reducedFrame, Size(frame.cols * 0.25, frame.rows * 0.25));

        vector<thread> threads;
        vector<Rect> updatedBboxes(trackedObjects.size());

        for (int i = 0; i < trackedObjects.size(); ++i) {
            threads.emplace_back([i, &trackedObjects, &frame, &updatedBboxes] {
                bool ok = trackedObjects[i].tracker->update(frame, updatedBboxes[i]);
                if (!ok) {
                    cerr << "Tracking failed for object " << i << endl;
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // Draw bounding boxes on the original frame
        drawBoundingBoxes(frame, updatedBboxes);

        // Calculate FPS
        double endTime = getTickCount();
        double elapsedTime = (endTime - startTime) / getTickFrequency();
        totalFrames++;
        if (elapsedTime >= 1.0) {
            double fps = totalFrames / elapsedTime;
            fpsValues.push_back(fps); // Store FPS value

            // Write FPS data to CSV file
            csvFile << fps << endl;

            // Reset for the next interval
            startTime = endTime;
            totalFrames = 0;
        }

        // Draw average FPS on the original frame
        if (!fpsValues.empty()) {
            double averageFps = accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
            string fpsText = "FPS: " + to_string(static_cast<int>(averageFps));
            putText(frame, fpsText, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }

        // Prepare the display frame (resizing for display purposes only)
        resize(frame, displayFrame, Size(displayWidth, displayHeight));

        // Store the frame in its original size with FPS information
        frames.push_back(frame.clone());

        // Display the resized frame
        imshow("Tracking", displayFrame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Close CSV file
    csvFile.close();

    // Release video writer and capture
    cap.release();
    destroyAllWindows();

    // Save frames as a video in original size
    if (!frames.empty()) {
        VideoWriter videoWriter("tracked_output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frames[0].size());
        if (!videoWriter.isOpened()) {
            cerr << "Error opening video writer\n";
            return 1;
        }

        for (const Mat& frm : frames) {
            videoWriter.write(frm);
        }

        videoWriter.release();
    } else {
        cerr << "No frames to write.\n";
    }

    // Calculate and print average FPS
    if (!fpsValues.empty()) {
        double averageFps = accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
        cout << "Average FPS: " << averageFps << endl;
    } else {
        cout << "No FPS values recorded." << endl;
    }

    return 0;
}
