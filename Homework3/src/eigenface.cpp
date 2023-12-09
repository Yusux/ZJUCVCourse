#include "eigenface.hpp"
#include "face.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>

using namespace cv;

EigenFace::EigenFace(bool is_train, std::vector<String> paths, double threshold) {
    try {
        // if is_train is true, train the model
        if (is_train) {
            // get face images
            std::vector<std::pair<Face, String> > faces;
            getEigenFace_(paths, faces);
            // train the model
            train_(faces, threshold);        
        } else { // if is_train is false, load the model
            // load the model
            // check if the model exists
            checkFile("eigenface.yml");
            FileStorage fs("eigenface.yml", FileStorage::READ);
            fs["convert_mat"] >> convert_mat_;
            fs["mean_face"] >> mean_face_;
            FileNode eigen_faces = fs["eigen_faces"];
            for (FileNodeIterator it = eigen_faces.begin(); it != eigen_faces.end(); it++) {
                FileNode eigen_face_node = *it;
                Mat eigen_face;
                String label;
                eigen_face_node["eigen_face"] >> eigen_face;
                eigen_face_node["label"] >> label;
                eigen_faces_.push_back(std::make_pair(eigen_face, label));
            }
            fs.release();
        }
    } catch (Exception& e) {
        throw e;
    }
}

EigenFace::EigenFace(const EigenFace& eigenface) {
    eigenface.convert_mat_.copyTo(convert_mat_);
    eigenface.mean_face_.copyTo(mean_face_);
    eigen_faces_ = eigenface.eigen_faces_;
}

EigenFace::~EigenFace() {
    // save the model
    FileStorage fs("eigenface.yml", FileStorage::WRITE);
    fs << "convert_mat" << convert_mat_;
    fs << "mean_face" << mean_face_;
    fs << "eigen_faces" << "[";
    for (int i = 0; i < eigen_faces_.size(); i++) {
        fs << "{";
        fs << "eigen_face" << eigen_faces_[i].first;
        fs << "label" << eigen_faces_[i].second;
        fs << "}";
    }
    fs << "]";
    fs.release();
}

void EigenFace::getEigenFace_(std::vector<String> &paths,
                              std::vector<std::pair<Face, String> > &faces) {
    // get face image paths (and eye position) paths
    if (paths.size() == 0) {
        throw Exception(-215, "No face image path provided", "EigenFace::EigenFace", __FILE__, __LINE__);
    }
    std::vector<String> face_image_paths;
    std::vector<String> eye_position_paths;
    checkFolder(paths[0], face_image_paths, true);
    if (paths.size() > 1) {
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
        std::cout << "face_image_path: " << face_image_path << std::endl;
        std::cout << "eye_position_path: " << eye_position_path << std::endl;

        // get label like "s31" in
        // "../Images/attface/att-face/s31/9.pgm"
        String parent_dir = face_image_path.substr(0, face_image_path.find_last_of("/\\"));
        String label = parent_dir.substr(parent_dir.find_last_of("/\\") + 1);

        // add face and string pair to faces
        faces.push_back(std::make_pair(Face(face_image_path, eye_position_path), label));
    }

    std::cout << "faces.size(): " << faces.size() << std::endl;

    // calculate average face
    if (faces.size() == 0) {
        throw Exception(-215, "No face image provided", "EigenFace::EigenFace", __FILE__, __LINE__);
    }
    Size face_image_size = faces[0].first.getViewedFaceImage().size();

    std::cout << "face_image_size: " << face_image_size << std::endl;

    Mat average_face = Mat::zeros(face_image_size, CV_32FC1);
    for (int i = 0; i < faces.size(); i++) {
        average_face += faces[i].first.getViewedFaceImage();
    }
    average_face /= faces.size();
    mean_face_ = average_face.reshape(1, 1);

    std::cout << "average_face.size(): " << average_face.size() << std::endl;
}

void EigenFace::train_(std::vector<std::pair<Face, String> > &faces, double threshold) {
    // USE SVG TO CALCULATE EIGENVALUES AND EIGENVECTORS
    Size face_image_size = faces[0].first.getViewedFaceImage().size();
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
    std::cout << "X.size(): " << X.size() << std::endl;

    // use SVD to calculate U, S, V
    Mat U, S, Vt;
    SVD::compute(X, S, U, Vt);
    std::cout << "U.size(): " << U.size() << std::endl;
    std::cout << "S.size(): " << S.size() << std::endl;
    std::cout << "Vt.size(): " << Vt.size() << std::endl;

    // square S to get eigenvalues of X^T * X
    for (int i = 0; i < S.rows; i++) {
        S.at<float>(i, 0) = S.at<float>(i, 0) * S.at<float>(i, 0);
    }
    // calculate the number of eigenvalues to use
    int eigenvalues_num = 0;
    double eigenvalues_sum = 0;
    for (int i = 0; i < S.rows; i++) {
        eigenvalues_sum += S.at<float>(i, 0);
    }
    double eigenvalues_sum_tmp = 0;
    for (int i = 0; i < S.rows; i++) {
        eigenvalues_sum_tmp += S.at<float>(i, 0);
        if (eigenvalues_sum_tmp / eigenvalues_sum >= threshold) {
            eigenvalues_num = i + 1;
            break;
        }
    }

    // get the first eigenvalues_num eigenvectors
    // Since the covariance matrix is corresponding to the X^T * X
    // and the eigenvectors of X^T * X is V
    Mat eigenvalues = Mat::zeros(eigenvalues_num, 1, CV_32FC1);
    Mat eigenvectors = Mat::zeros(eigenvalues_num, face_image_size.area(), CV_32FC1);
    for (int i = 0; i < eigenvalues_num; i++) {
        eigenvalues.at<float>(i, 0) = S.at<float>(i, 0);
        Vt.row(i).copyTo(eigenvectors.row(i));
    }
    std::cout << "eigenvalues.size(): " << eigenvalues.size() << std::endl;
    
    std::ofstream eigenvalues_file("eigenvalues.txt");
    for (int i = 0; i < eigenvalues.rows; i++) {
        eigenvalues_file << eigenvalues.at<float>(i, 0) << std::endl;
    }
    eigenvalues_file.close();

    // get convert mat
    convert_mat_ = eigenvectors;
    std::cout << "convert_mat_.size(): " << convert_mat_.size() << std::endl;
    std::ofstream SVD_convert_mat_file("SVD_convert_mat.txt");
    for (int i = 0; i < convert_mat_.rows; i++) {
        for (int j = 0; j < convert_mat_.cols; j++) {
            SVD_convert_mat_file << convert_mat_.at<float>(i, j) << " ";
        }
        SVD_convert_mat_file << std::endl;
    }
    SVD_convert_mat_file.close();

    // get face images in the new space
    for (int i = 0; i < faces.size(); i++) {
        std::cout << "i: " << i << std::endl;
        // face image in the new space
        Mat tmp_face_image;
        faces[i].first.getViewedFaceImage().convertTo(tmp_face_image, CV_32FC1);
        tmp_face_image -= mean_face_;
        std::cout << "tmp_face_image.size(): " << tmp_face_image.size() << std::endl;
        std::cout << "convert_mat_.cols: " << convert_mat_.cols << std::endl;
        std::cout << "tmp_face_image.rows: " << tmp_face_image.rows << std::endl;
        Mat face_image_new_space = convert_mat_ * tmp_face_image.t();
        std::cout << "face_image_new_space.size(): " << face_image_new_space.size() << std::endl;
        // add face image in the new space to face images
        eigen_faces_.push_back(std::make_pair(face_image_new_space, faces[i].second));
    }
}

Mat EigenFace::getConvertMat() {
    return convert_mat_;
}

Mat EigenFace::getMeanFace() {
    return mean_face_;
}

std::vector<std::pair<Mat, String> > EigenFace::getFaceImages() {
    return eigen_faces_;
}

String EigenFace::recognize(String face_image_path, String eye_position_path) {
    // get the Face object
    Face face(face_image_path, eye_position_path);

    Mat face_image_to_recognize;
    face.getViewedFaceImage().convertTo(face_image_to_recognize, CV_32FC1);
    face_image_to_recognize -= mean_face_;

    // face image in the new space
    Mat face_image_new_space = convert_mat_ * face_image_to_recognize.t();

    // calculate the distance between face_image_new_space and each face image in the new space
    double min_distance = DBL_MAX;
    String label = "";
    for (int i = 0; i < eigen_faces_.size(); i++) {
        double distance = norm(face_image_new_space, eigen_faces_[i].first);
        if (distance < min_distance) {
            min_distance = distance;
            label = eigen_faces_[i].second;
        }
    }

    return label;
}

Mat EigenFace::reconstruct(String face_image_path, String eye_position_path) {
    // get the Face object
    Face face(face_image_path, eye_position_path);

    Mat face_image_to_reconstruct;
    face.getViewedFaceImage().convertTo(face_image_to_reconstruct, CV_32FC1);
    face_image_to_reconstruct -= mean_face_;

    // face image in the new space
    Mat face_image_new_space = convert_mat_ * face_image_to_reconstruct.t();
    // reconstruct face image in the new space
    Mat face_image_reconstructed = convert_mat_.t() * face_image_new_space;
    // add mean face to face_image_reconstructed
    face_image_reconstructed = face_image_reconstructed.t() + mean_face_;

    // recover face image from [1 x 9900] to [110 x 90]
    face_image_reconstructed = face_image_reconstructed.reshape(1, 110);

    return face_image_reconstructed;
}
