#ifndef __EIGENFACE_HPP__
#define __EIGENFACE_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "../face.hpp"

class EigenFace {
public:
    EigenFace(std::vector<cv::String> paths, int pca_dim = 10);
    ~EigenFace();

    double recognize(cv::String face_images_path, cv::String eye_positions_path = "");

private:
    cv::Mat convert_mat_;
    cv::Mat mean_face_;
    std::vector<std::pair<cv::Mat, cv::String> > eigen_faces_;
    
    void getFaces_(std::vector<cv::String> &paths,
                   std::vector<std::pair<Face, cv::String> > &faces,
                   int need_odd);
    void train_(std::vector<std::pair<Face, cv::String> > &faces, int pca_dim = 10);
};

#endif