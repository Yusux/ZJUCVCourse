#include "eigenface_mod.hpp"
#include "../face.hpp"
#include "../utils.hpp"

using namespace cv;

EigenFace::EigenFace(std::vector<String> paths, int pca_dim) {
    try {
        // get face images
        std::vector<std::pair<Face, String> > faces;
        getFaces_(paths, faces, 0);
        // train the model
        train_(faces, pca_dim);
    } catch (Exception& e) {
        throw e;
    }
}

EigenFace::~EigenFace() {
}

void EigenFace::getFaces_(std::vector<String> &paths,
                              std::vector<std::pair<Face, String> > &faces,
                              int need_odd) {
    // get face image paths (and eye position) paths
    if (paths.size() == 0) {
        throw Exception(-215, "No face image path provided", "EigenFace::getFaces_", __FILE__, __LINE__);
    }
    std::vector<String> face_image_paths;
    std::vector<String> eye_position_paths;
    checkFolder(paths[0], face_image_paths, true);
    if (paths.size() > 1 && paths[1] != "") {
        checkFolder(paths[1], eye_position_paths, true);
    }

    // read all the face images (and eye positions)
    // if eye position exists, then the two vectors
    // should have the same size and the same order
    for (int i = 0; i < face_image_paths.size(); i++) {
        // init face image (and eye position) path
        String face_image_path = face_image_paths[i];
        String eye_position_path = "";
        if (eye_position_paths.size() > 0) {
            eye_position_path = eye_position_paths[i];
        }

        // get label like "s31" in
        // "../Images/attface/att-face/s31/9.pgm"
        String parent_dir = face_image_path.substr(0, face_image_path.find_last_of("/\\"));
        String label = parent_dir.substr(parent_dir.find_last_of("/\\") + 1);
        String idx_string = face_image_path.substr(face_image_path.find_last_of("/\\") + 1,
            face_image_path.find_last_of(".") - face_image_path.find_last_of("/\\") - 1);
        int idx = std::stoi(idx_string);
        // only use the odd/even index images in each folder
        if ((idx % 2) == need_odd) {
            continue;
        }

        // add face and string pair to faces and original_face_images_
        Face face(face_image_path, eye_position_path);
        faces.push_back(std::make_pair(face, label));
    }
}

void EigenFace::train_(std::vector<std::pair<Face, String> > &faces, int pca_dim) {
    // calculate average face
    if (faces.size() == 0) {
        throw Exception(-215, "No face image provided", "EigenFace::train_", __FILE__, __LINE__);
    }
    Size face_image_size = faces[0].first.getViewedFaceImage().size();

    mean_face_ = Mat::zeros(face_image_size, CV_32FC1);
    for (int i = 0; i < faces.size(); i++) {
        mean_face_ += faces[i].first.getViewedFaceImage();
    }
    mean_face_ /= faces.size();

    // USE SVG TO CALCULATE EIGENVALUES AND EIGENVECTORS
    face_image_size = faces[0].first.getViewedFaceImage().size();
    // calculate X
    Mat X = Mat::zeros(faces.size(), face_image_size.area(), CV_32FC1);
    for (int i = 0; i < faces.size(); i++) {
        // difference between face image and average face
        Mat tmp_face_image;
        faces[i].first.getViewedFaceImage().convertTo(tmp_face_image, CV_32FC1);
        Mat face_image_diff = tmp_face_image - mean_face_;
        // add face image diff to X
        face_image_diff.reshape(1, 1).copyTo(X.row(i));
    }

    // use SVD to calculate S, V
    Mat _U, S, Vt;
    SVD::compute(X, S, _U, Vt);

    // square S to get eigenvalues of X^T * X
    for (int i = 0; i < S.rows; i++) {
        S.at<float>(i, 0) = S.at<float>(i, 0) * S.at<float>(i, 0);
    }

    // get the first pca_dim eigenvectors
    // Since the covariance matrix is corresponding to the X^T * X
    // and the eigenvectors of X^T * X is V
    Mat eigenvalues = Mat::zeros(pca_dim, 1, CV_32FC1);
    Mat eigenvectors = Mat::zeros(pca_dim, face_image_size.area(), CV_32FC1);
    for (int i = 0; i < pca_dim; i++) {
        eigenvalues.at<float>(i, 0) = S.at<float>(i, 0);
        Vt.row(i).copyTo(eigenvectors.row(i));
    }

    // get convert mat
    convert_mat_ = eigenvectors;

    // get face images in the new space
    for (int i = 0; i < faces.size(); i++) {
        // face image in the new space
        Mat tmp_face_image;
        faces[i].first.getViewedFaceImage().convertTo(tmp_face_image, CV_32FC1);
        tmp_face_image -= mean_face_;
        Mat face_image_new_space = convert_mat_ * tmp_face_image.t();

        // add face image in the new space to face images
        eigen_faces_.push_back(std::make_pair(face_image_new_space, faces[i].second));
    }
}

double EigenFace::recognize(String face_images_path, String eye_positions_path) {
    // get the Face object
    std::vector<cv::String> paths = {face_images_path, eye_positions_path};
    std::vector<std::pair<Face, cv::String> > faces;
    getFaces_(paths, faces, 1);

    // count the number of correct predictions
    int correct_count = 0;

    for (auto face_pair : faces) {
        Face face = face_pair.first;
        String label = face_pair.second;

        Mat face_image_to_recognize;
        face.getViewedFaceImage().convertTo(face_image_to_recognize, CV_32FC1);
        face_image_to_recognize -= mean_face_;

        // face image in the new space
        Mat face_image_new_space = convert_mat_ * face_image_to_recognize.t();

        // calculate the distance between face_image_new_space and each face image in the new space
        double min_distance = DBL_MAX;
        int min_index = -1;
        String assuming_label = "";
        for (int i = 0; i < eigen_faces_.size(); i++) {
            double distance = norm(face_image_new_space, eigen_faces_[i].first);
            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
                assuming_label = eigen_faces_[i].second;
            }
        }

        // if the label is correct, then correct_count++
        if (assuming_label == label) {
            correct_count++;
        }
    }

    // return the accuracy
    return (double)correct_count / faces.size();
}
