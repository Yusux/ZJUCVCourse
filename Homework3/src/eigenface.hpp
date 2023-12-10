#ifndef __EIGENFACE_HPP__
#define __EIGENFACE_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "face.hpp"

class EigenFace {
public:
    /*
     * Constructor of EigenFace class.
     * @param is_train: whether to train the model.
     * @param paths: paths of face images.
     * @param inner: whether to use inner face images.
     * @param threshold: threshold of eigenface (energy ratio).
     */
    EigenFace(bool is_train, std::vector<cv::String> paths, bool inner = false, double threshold = 0.99);

    /*
     * Copy constructor of EigenFace class.
     * @param eigenface: another EigenFace object.
     */
    EigenFace(const EigenFace& eigenface);

    /*
     * Destructor of EigenFace class,
     * including saving the model to config file.
     */
    ~EigenFace();

    /*
     * Get convert matrix.
     * @return convert matrix.
     */
    cv::Mat getConvertMat() const;

    /*
     * Get mean face.
     * @return mean face.
     */
    cv::Mat getMeanFace() const;

    /*
     * Get eigen faces.
     * @return eigen faces.
     */
    std::vector<std::pair<cv::Mat, cv::String> > getFaceImages() const;

    /*
     * Recognize face image.
     * @param face_image_path: path of face image.
     * @param eye_position_path: path of eye position.
     * @return label of face image.
     */
    cv::String recognize(cv::String face_image_path, cv::String eye_position_path = "");

    /*
     * Reconstruct face image.
     * @param face_image_path: path of face image.
     * @param eye_position_path: path of eye position.
     * @return reconstructed face image.
     */
    cv::Mat reconstruct(cv::String face_image_path, cv::String eye_position_path = "");

private:
    bool inner_;                // whether to output inner product
    cv::String config_path_;    // path of config file
    cv::Mat convert_mat_;       // convert matrix
    cv::Mat mean_face_;         // mean face
    // eigen faces converted by convert matrix
    std::vector<std::pair<cv::Mat, cv::String> > eigen_faces_;
    // original face images
    std::vector<cv::Mat> original_face_images_;

    /*
     * Get face images.
     * @param paths: paths of face images.
     * @param faces: face images.
     */
    void getFaces_(std::vector<cv::String> &paths,
                   std::vector<std::pair<Face, cv::String> > &faces);

    /*
     * Train the model.
     * @param faces: face images.
     * @param threshold: threshold of eigenface (energy ratio).
     */
    void train_(std::vector<std::pair<Face, cv::String> > &faces, double threshold);
};

#endif