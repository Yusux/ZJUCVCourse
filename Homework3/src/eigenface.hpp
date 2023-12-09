#ifndef __EIGENFACE_HPP__
#define __EIGENFACE_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "face.hpp"

class EigenFace {
public:
    EigenFace(bool is_train, std::vector<cv::String> paths, bool inner = false, double threshold = 0.99);
    EigenFace(const EigenFace& eigenface);
    ~EigenFace();

    cv::Mat getConvertMat();
    cv::Mat getMeanFace();
    std::vector<std::pair<cv::Mat, cv::String> > getFaceImages();
    cv::String recognize(cv::String face_image_path, cv::String eye_position_path = "");
    cv::Mat reconstruct(cv::String face_image_path, cv::String eye_position_path = "");

private:
    bool inner_;
    cv::String config_path_;
    cv::Mat convert_mat_;
    cv::Mat mean_face_;
    std::vector<std::pair<cv::Mat, cv::String> > eigen_faces_;
    std::vector<cv::Mat> original_face_images_;
    
    void getFaces_(std::vector<cv::String> &paths,
                       std::vector<std::pair<Face, cv::String> > &faces);
    void train_(std::vector<std::pair<Face, cv::String> > &faces, double threshold);
};

#endif