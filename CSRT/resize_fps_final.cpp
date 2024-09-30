#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <thread>
#include <iostream>
#include <numeric> // For std::accumulate
#include <chrono>

using namespace cv;
using namespace std;

struct TrackedObject {
    Rect bbox;
    Ptr<Tracker> tracker;
};

vector<Rect> selectBboxes(const Mat& frame) {
    vector<Rect> bboxes;
    Mat displayFrame;
    Size newSize(960, 540);
    resize(frame, displayFrame, newSize);

    while (true) {
        Rect bbox = selectROI("Select object (Press 'c' to cancel)", displayFrame);
        if (bbox.width > 0 && bbox.height > 0) {
            bbox.x = static_cast<int>(bbox.x * frame.cols / 960);
            bbox.y = static_cast<int>(bbox.y * frame.rows / 540);
            bbox.width = static_cast<int>(bbox.width * frame.cols / 960);
            bbox.height = static_cast<int>(bbox.height * frame.rows / 540);
            bboxes.push_back(bbox);
        } else {
            break;
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
    // Step 1: Resize the video using FFmpeg
    system("ffmpeg -i DJI_0874.mp4 -vf scale=960:540 resized_video.MOV");

    // Step 2: Process the resized video
    VideoCapture cap("resized_video.MOV");
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
        obj.tracker = TrackerCSRT::create(params);
        obj.tracker->init(frame, bbox);
        trackedObjects.push_back(obj);
    }

    vector<Mat> frames;
    int totalFrames = 0;
    auto startTime = chrono::high_resolution_clock::now();
    vector<double> fpsValues;
    
    const Size displaySize(960, 540);

    while (cap.read(frame)) {
        Mat reducedFrame, displayFrame;
        resize(frame, reducedFrame, displaySize); // Ensure frame is at 960x540

        vector<thread> threads(trackedObjects.size());
        vector<Rect> updatedBboxes(trackedObjects.size());

        for (int i = 0; i < trackedObjects.size(); ++i) {
            threads[i] = thread([i, &trackedObjects, &reducedFrame, &updatedBboxes] {
                bool ok = trackedObjects[i].tracker->update(reducedFrame, updatedBboxes[i]);
                if (!ok) {
                    cerr << "Tracking failed for object " << i << endl;
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        drawBoundingBoxes(reducedFrame, updatedBboxes);

        totalFrames++;
        auto now = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = now - startTime;

        if (elapsed.count() >= 1.0) { // Update FPS every second
            double fps = totalFrames / elapsed.count();
            fpsValues.push_back(fps);

            // Reset counters
            startTime = chrono::high_resolution_clock::now();
            totalFrames = 0;
        }

        if (!fpsValues.empty()) {
            double averageFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
            putText(reducedFrame, "FPS: " + to_string(static_cast<int>(averageFps)), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }

        resize(reducedFrame, displayFrame, displaySize);
        frames.push_back(reducedFrame.clone());
        imshow("Tracking", displayFrame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    if (!frames.empty()) {
        VideoWriter videoWriter("tracked_output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, displaySize);
        if (!videoWriter.isOpened()) {
            cerr << "Error opening video writer\n";
            return 1;
        }

        for (const Mat& frm : frames) {
            videoWriter.write(frm);
        }
    } else {
        cerr << "No frames to write.\n";
    }

    if (!fpsValues.empty()) {
        double averageFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
        cout << "Average FPS: " << averageFps << endl;
    } else {
        cout << "No FPS values recorded." << endl;
    }

    return 0;
}
