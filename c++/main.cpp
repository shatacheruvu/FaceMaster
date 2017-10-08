//
// Created by Shata Cheruvu at Autonxt Automation Pvt. Ltd. on 08/10/17.
//

#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#define HUMAN_FACE_CLASSIFIER "../classifiers/haarcascade_frontalface_default.xml"

using namespace cv;
using namespace std;

int main()
{
    CascadeClassifier faceCascade;
    Mat videoFrame;
    vector<Rect> humanFaces;
    VideoCapture stream(0);

    if(!faceCascade.load(HUMAN_FACE_CLASSIFIER)) {
        cout << "Cannot load human face classifier" << flush;
        return 0;
    }
    if(!stream.isOpened()) {
        cout << "Cannot open camera" << flush;
        return 0;
    }

    while (true) {
        stream.read(videoFrame);
        faceCascade.detectMultiScale(videoFrame, humanFaces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (auto &humanFace : humanFaces) {
            Point pointOne(humanFace.x, humanFace.y);
            Point pointTwo(humanFace.x+humanFace.width, humanFace.y+humanFace.height);
            rectangle(videoFrame, pointOne, pointTwo, Scalar(255,255,255), 2, LINE_8);
        }
        imshow("camera feed", videoFrame);
        if (waitKey(1) >= 0)
            break;
    }
    return 0;
}