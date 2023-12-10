#include "eigenface.hpp"
#include "face.hpp"
#include "utils.hpp"

using namespace cv;

EigenFace::EigenFace(bool is_train, std::vector<String> paths, bool inner, double threshold) {
    // get config path
    if (paths.size() == 0) {
        throw Exception(-215, "No config path provided", "EigenFace::EigenFace", __FILE__, __LINE__);
    }
    config_path_ = paths[0];
    paths.erase(paths.begin());

    // check threshold whether is valid
    if (is_train && (threshold <= 0 || threshold > 1)) {
        throw Exception(-215, "Invalid threshold", "EigenFace::EigenFace", __FILE__, __LINE__);
    }

    // set inner noting whether to output inner image
    inner_ = inner;

    try {
        // if is_train is true, train the model
        if (is_train) {
            // get face images
            std::vector<std::pair<Face, String> > faces;
            getFaces_(paths, faces);
            // train the model
            train_(faces, threshold);
        } else { // if is_train is false, load the model
            // load the model
            // check if the model exists
            checkFile(config_path_);
            FileStorage fs(config_path_, FileStorage::READ);
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
            FileNode original_face_images = fs["original_face_images"];
            for (FileNodeIterator it = original_face_images.begin(); it != original_face_images.end(); it++) {
                FileNode original_face_image_node = *it;
                Mat original_face_image;
                original_face_image_node >> original_face_image;
                original_face_images_.push_back(original_face_image);
            }
            fs.release();
        }
    } catch (Exception& e) {
        throw e;
    }
}

EigenFace::EigenFace(const EigenFace& eigenface) {
    inner_ = eigenface.inner_;
    config_path_ = eigenface.config_path_;
    eigenface.convert_mat_.copyTo(convert_mat_);
    eigenface.mean_face_.copyTo(mean_face_);
    eigen_faces_ = eigenface.eigen_faces_;
    original_face_images_ = eigenface.original_face_images_;
}

EigenFace::~EigenFace() {
    // save the model
    FileStorage fs(config_path_, FileStorage::WRITE);
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
    fs << "original_face_images" << "[";
    for (int i = 0; i < original_face_images_.size(); i++) {
        fs << original_face_images_[i];
    }
    fs << "]";
    fs.release();
}

void EigenFace::getFaces_(std::vector<String> &paths,
                              std::vector<std::pair<Face, String> > &faces) {
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
        String label = face_image_path.substr(parent_dir.find_last_of("/\\") + 1);

        // add face and string pair to faces and original_face_images_
        Face face(face_image_path, eye_position_path);
        faces.push_back(std::make_pair(face, label));
        original_face_images_.push_back(face.getFaceImage());
    }
}

void EigenFace::train_(std::vector<std::pair<Face, String> > &faces, double threshold) {
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
    // to ensure that the program can run normally
    // according to the test requirement: "need 100 PCs"
    if (eigenvalues_num < 100) {
        eigenvalues_num = 100;
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

    // if inner_ is true, output the inner image
    if (inner_) {
        // every 2 images are separated by 5 pixels
        Mat eigen_faces_image = Mat::zeros(110, 90 * 11 + 50, CV_32FC1);
        Point2i tl(0, 0);
        // show mean face
        Mat mean_face = mean_face_.clone();
        mean_face = mean_face.reshape(1, 110);
        normalize(mean_face, mean_face, 0, 255, NORM_MINMAX);
        mean_face.copyTo(eigen_faces_image(Rect(tl, Size(90, 110))));
        // show first 10 eigen faces
        for (int i = 0; i < 10; i++) {
            tl.x += 90 + 5;
            Mat eigen_face = convert_mat_.row(i).clone();
            eigen_face = eigen_face.reshape(1, 110);
            normalize(eigen_face, eigen_face, 0, 255, NORM_MINMAX);
            eigen_face.copyTo(eigen_faces_image(Rect(tl, Size(90, 110))));
        }

        // save the inner image
        imwrite("inner/train_10_eigen_faces.jpg", eigen_faces_image);
    }
}

Mat EigenFace::getConvertMat() const {
    return convert_mat_.clone();
}

Mat EigenFace::getMeanFace() const {
    return mean_face_.clone();
}

std::vector<std::pair<Mat, String> > EigenFace::getFaceImages() const {
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
    int min_index = -1;
    String label = "";
    for (int i = 0; i < eigen_faces_.size(); i++) {
        double distance = norm(face_image_new_space, eigen_faces_[i].first);
        if (distance < min_distance) {
            min_distance = distance;
            min_index = i;
            label = eigen_faces_[i].second;
        }
    }

    // if inner_ is true, output the inner image
    if (inner_) {
        // get the face image
        Mat face_image = face.getFaceImage();
        // get the most similar face image
        Mat most_similar = original_face_images_[min_index];

        // concat the two images
        Mat concat_image = Mat::zeros(135, 90 * 2 + 5, CV_32FC1);
        face_image.copyTo(concat_image(Rect(Point(0, 0), face_image.size())));
        most_similar.copyTo(concat_image(Rect(Point(90 + 5, 0), most_similar.size())));

        // overlay the label on the face image
        putText(concat_image, label, Point(10, 125), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255), 1.2);

        // save the inner image
        imwrite("inner/recognize_result.jpg", concat_image);
    }

    return label;
}

Mat EigenFace::reconstruct(String face_image_path, String eye_position_path) {
    // get the Face object
    Face face(face_image_path, eye_position_path);

    // face image to reconstruct
    Mat face_image_to_reconstruct;
    face.getViewedFaceImage().convertTo(face_image_to_reconstruct, CV_32FC1);
    face_image_to_reconstruct -= mean_face_;

    // face image in the new space
    Mat face_image_new_space = convert_mat_ * face_image_to_reconstruct.t();
    // reconstruct face image in the new space
    Mat face_image_reconstructed = convert_mat_.t() * face_image_new_space;
    // add mean face to face_image_reconstructed
    face_image_reconstructed = face_image_reconstructed.t() + mean_face_;

    // convert face_image_reconstructed to CV_8UC1
    face_image_reconstructed.convertTo(face_image_reconstructed, CV_8UC1);

    // recover face image from [1 x 9900] to [110 x 90]
    face_image_reconstructed = face_image_reconstructed.reshape(1, 110);

    // if inner_ is true, output the inner image
    if (inner_) {
        // set the number of PCs
        std::vector<int> pcs = {10, 25, 50, 100};
        
        // create the inner image
        Mat inner_image = Mat::zeros(110, 90 * 5 + 5 * 4, CV_8UC1);
        Point2i tl(0, 0);

        // show the original face image
        face.getFaceImage().copyTo(inner_image(Rect(tl, Size(90, 110))));

        for (auto pc : pcs) {
            tl.x += 90 + 5;
            // get the convert mat with pc PCs
            Mat convert_mat = convert_mat_.rowRange(0, pc);

            // get the face image to reconstruct
            Mat face_image_to_reconstruct;
            face.getViewedFaceImage().convertTo(face_image_to_reconstruct, CV_32FC1);
            face_image_to_reconstruct -= mean_face_;

            // face image in the new space
            Mat face_image_new_space = convert_mat * face_image_to_reconstruct.t();
            // reconstruct face image in the new space
            Mat face_image_reconstructed = convert_mat.t() * face_image_new_space;
            // add mean face to face_image_reconstructed
            face_image_reconstructed = face_image_reconstructed.t() + mean_face_;

            // convert face_image_reconstructed to CV_8UC1
            face_image_reconstructed.convertTo(face_image_reconstructed, CV_8UC1);

            // recover face image from [1 x 9900] to [110 x 90]
            face_image_reconstructed = face_image_reconstructed.reshape(1, 110);

            // copy the face image to inner image
            face_image_reconstructed.copyTo(inner_image(Rect(tl, Size(90, 110))));
        }

        // save the inner image
        imwrite("inner/reconstruct_result.jpg", inner_image);
    }

    return face_image_reconstructed;
}
