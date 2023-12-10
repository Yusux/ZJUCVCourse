#include "face.hpp"
#include "utils.hpp"
#include <fstream>
#include <json/json.h>

using namespace cv;

static const int FACE_EYE_DISTANCE = 36;
static const Point2f LEFT_EYE_POSITION = Point2f(28, 45);
static const Size FACE_IMAGE_SIZE = Size(90, 110);

CascadeClassifier Face::face_cascade_ = CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml");
CascadeClassifier Face::eyes_cascade_ = CascadeClassifier("haarcascade/haarcascade_eye_tree_eyeglasses.xml");

Face::Face(String face_image_path, String eye_position_path) {
    // read face image
    face_image_ = imread(face_image_path, IMREAD_GRAYSCALE);

    try {
        // detect eyes or read eye positions
        if (eye_position_path == "") {
            // if eye position path is not specified, detect eyes
            detectEyes();
        } else {
            // if eye position path is specified, read eye positions
            readEyePosition(eye_position_path);
        }
        // transform face image
        transformFaceImage();
        // normalize face image by histogram equalization and stretch
        normalizeFaceImage();
        // view face image into a vector
        viewFaceImage();
    } catch (Exception& e) {
        throw e;
    }
}

Face::Face(const Face& face) {
    face.face_image_.copyTo(face_image_);
    face.viewed_face_image_.copyTo(viewed_face_image_);
    left_eye_ = face.left_eye_;
    right_eye_ = face.right_eye_;
}

Face::~Face() {
}

Rect Face::detectFace() {
    // detect face
    std::vector<Rect> faces;
    face_cascade_.detectMultiScale(face_image_, faces, 1.3, 5, CASCADE_DO_CANNY_PRUNING, face_image_.size()/4);

    Mat tmp = face_image_.clone();
    for (int i = 0; i < faces.size(); i++) {
        rectangle(tmp, faces[i], Scalar(255, 255, 255));
    }
    imwrite("face.jpg", tmp);

    // if no face detected, throw an exception
    if (faces.size() == 0) {
        throw Exception(-215, "No face detected", "Face::detectFace", __FILE__, __LINE__);
    }

    // if more than one face detected, throw an exception
    if (faces.size() > 1) {
        throw Exception(-215, "More than one face detected", "Face::detectFace", __FILE__, __LINE__);
    }

    // return the detected face
    return faces[0];
}

void Face::detectEyes() {
    // detect face
    Rect face = detectFace();

    // detect eyes
    std::vector<Rect> eyes;
    eyes_cascade_.detectMultiScale(face_image_(face), eyes, 1.2, 3, CASCADE_DO_CANNY_PRUNING, face_image_(face).size()/8);

    Mat tmp = face_image_(face).clone();
    for (int i = 0; i < eyes.size(); i++) {
        rectangle(tmp, eyes[i], Scalar(255, 255, 255));
    }
    imwrite("eyes.jpg", tmp);

    // if no eye detected, throw an exception
    if (eyes.size() < 2) {
        throw Exception(-215, "Less than two eyes detected", "Face::detectEyes", __FILE__, __LINE__);
    }

    // if more than two eyes detected, throw an exception
    if (eyes.size() > 2) {
        throw Exception(-215, "More than two eyes detected", "Face::detectEyes", __FILE__, __LINE__);
    }

    // two eyes detected, assign them to left eye and right eye
    if (eyes[0].x < eyes[1].x) {
        left_eye_ = getRectCenter(eyes[0]) + face.tl();
        right_eye_ = getRectCenter(eyes[1]) + face.tl();
    } else {
        left_eye_ = getRectCenter(eyes[1]) + face.tl();
        right_eye_ = getRectCenter(eyes[0]) + face.tl();
    }
}

void Face::readEyePosition(String eye_position_path) {
    // read eye position from json file
    Json::Value eye_position;
    std::ifstream fin(eye_position_path);
    if (!fin.is_open()) {
        throw Exception(-215, "Cannot open eye position file", "Face::Face", __FILE__, __LINE__);
    }
    fin >> eye_position;

    // check if the json file is valid
    if (!eye_position.isMember("centre_of_left_eye") || !eye_position.isMember("centre_of_right_eye")) {
        throw Exception(-215, "Invalid eye position file", "Face::Face", __FILE__, __LINE__);
    }

    // assign eye positions
    left_eye_.x = eye_position["centre_of_left_eye"][0].asInt();
    left_eye_.y = eye_position["centre_of_left_eye"][1].asInt();
    right_eye_.x = eye_position["centre_of_right_eye"][0].asInt();
    right_eye_.y = eye_position["centre_of_right_eye"][1].asInt();
}

void Face::transformFaceImage() {
    // Step 1. Prepare rotation matrix of the face image
    // so that the line connecting two eyes is horizontal.

    // get the angle between the line connecting two eyes
    // and the horizontal line
    double angle = atan2(right_eye_.y - left_eye_.y, right_eye_.x - left_eye_.x) * 180 / CV_PI;

    // get the center of the face image
    Point2f center(face_image_.cols / 2, face_image_.rows / 2);

    // get the rotation matrix
    Mat affine_mat = getRotationMatrix2D(center, angle, 1.0);


    // Step 2. Scale the face image so that the distance
    // between two eyes is a constant.

    // get the distance between two eyes
    double distance = norm(right_eye_ - left_eye_);

    // get the scale factor
    double scale = FACE_EYE_DISTANCE / distance;

    // add the scale factor to the rotation matrix
    affine_mat.at<double>(0,0) *= scale;
    affine_mat.at<double>(0,1) *= scale;
    affine_mat.at<double>(1,0) *= scale;
    affine_mat.at<double>(1,1) *= scale;


    // Step 3. move the face image so that
    // the left eye is at LEFT_EYE_POSITION

    // get the direction to move
    std::vector<Point2f> left_eye_vec = {left_eye_};
    transform(left_eye_vec, left_eye_vec, affine_mat);
    Point2f move_direction = LEFT_EYE_POSITION - left_eye_vec[0];

    // add the move direction to the rotation matrix
    affine_mat.at<double>(0,2) += move_direction.x;
    affine_mat.at<double>(1,2) += move_direction.y;


    // Step 4. apply affine transformation to the face image
    // and update the eye positions
    
    warpAffine(face_image_, face_image_, affine_mat, face_image_.size());
    std::vector<Point2f> eyes = {left_eye_, right_eye_};
    transform(eyes, eyes, affine_mat);
    left_eye_ = eyes[0];
    right_eye_ = eyes[1];


    // Step 5. crop the face image to FACE_IMAGE_SIZE

    // roi starts from (0, 0) and has size FACE_IMAGE_SIZE
    Rect roi(0, 0, FACE_IMAGE_SIZE.width, FACE_IMAGE_SIZE.height);

    // crop the face image
    face_image_ = face_image_(roi);
}

void Face::normalizeFaceImage() {
    // use histogram equalization and stretch to normalize the face image
    equalizeHist(face_image_, face_image_);
}

void Face::viewFaceImage() {
    // view the face image into a vector
    face_image_.copyTo(viewed_face_image_);
    viewed_face_image_ = viewed_face_image_.reshape(1, 1);
}

Mat Face::getFaceImage() const {
    return face_image_.clone();
}

Mat Face::getViewedFaceImage() const {
    return viewed_face_image_.clone();
}

Point2i Face::getLeftEye() const {
    return left_eye_;
}

Point2i Face::getRightEye() const {
    return right_eye_;
}
