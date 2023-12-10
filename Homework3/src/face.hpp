#ifndef __FACE_HPP__
#define __FACE_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect.hpp"

class Face {
public:
    /* Constructor of Face class.
     * @param face_image_path: path of face image.
     * @param eye_position_path: path of eye position.
     */
    Face(cv::String face_image_path, cv::String eye_position_path = "");

    /* Copy constructor of Face class.
     * @param face: another Face object.
     */
    Face(const Face& face);

    /* Destructor of Face class. */
    ~Face();

    /* Get face image.
     * @return face image.
     */
    cv::Mat getFaceImage() const;

    /* Get viewed face image.
     * @return viewed face image.
     */
    cv::Mat getViewedFaceImage() const;

    /* Get left eye center.
     * @return left eye center.
     */
    cv::Point2i getLeftEye() const;

    /* Get right eye center.
     * @return right eye center.
     */
    cv::Point2i getRightEye() const;

private:
    cv::Mat face_image_;                        // gray scale face image
    cv::Mat viewed_face_image_;                 // viewed face image
    cv::Point2i left_eye_;                      // left eye center
    cv::Point2i right_eye_;                     // right eye center
    static cv::CascadeClassifier face_cascade_; // face detector
    static cv::CascadeClassifier eyes_cascade_; // eyes detector

    /* Detect face.
     * @return face rectangle.
     */
    cv::Rect detectFace();

    /* Detect eyes, which will
     * modify left_eye_ and right_eye_.
     */
    void detectEyes();

    /* Read eye position from file, which will
     * modify left_eye_ and right_eye_.
     * @param eye_position_path: path of eye position.
     */
    void readEyePosition(cv::String eye_position_path);

    /* Transform face image to relatively fixed position
     * according to the position of eyes.
     */
    void transformFaceImage();

    /* Normalize face image by
     * histogram equalization and stretch.
     */
    void normalizeFaceImage();

    /* View face image into a vector. */
    void viewFaceImage();
};

#endif