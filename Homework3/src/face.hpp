#ifndef __FACE_HPP__
#define __FACE_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect.hpp"

class Face {
public:
    Face(cv::String face_image_path, cv::String eye_position_path = "");
    Face(const Face& face);
    ~Face();

    cv::Mat getFaceImage() const;
    cv::Mat getViewedFaceImage() const;
    cv::Point2i getLeftEye() const;
    cv::Point2i getRightEye() const;

private:
    cv::Mat face_image_;                        // gray scale face image
    cv::Mat viewed_face_image_;                 // viewed face image
    cv::Point2i left_eye_;                      // left eye center
    cv::Point2i right_eye_;                     // right eye center
    static cv::CascadeClassifier face_cascade_; // face detector
    static cv::CascadeClassifier eyes_cascade_; // eyes detector

    cv::Rect detectFace();
    void detectEyes();
    void readEyePosition(cv::String eye_position_path);
    void transformFaceImage();
    void normalizeFaceImage();
    void viewFaceImage();
};

#endif