#include "eigenface_mod.hpp"
#include "../utils.hpp"
#include <iostream>
#include <fstream>

using namespace cv;

int main(int argc, char** argv) {
    String face_images_path = "att/att-face";
    String eye_locations_path = "att/att-eye-location";
    try {
        std::ofstream fout("result.csv", std::ios::out);
        fout << "pca_dim,identity_rate" << std::endl;
        for (int pca_dim = 1; pca_dim < 40; pca_dim++) {
            std::cout << "pca_dim = " << pca_dim << std::endl;
            EigenFace eigenface({face_images_path, eye_locations_path}, pca_dim);
            double identity_rate = eigenface.recognize(face_images_path, eye_locations_path);
            fout << pca_dim << "," << identity_rate << std::endl;
        }
        for (int pca_dim = 40; pca_dim < 100; pca_dim += 10) {
            std::cout << "pca_dim = " << pca_dim << std::endl;
            EigenFace eigenface({face_images_path, eye_locations_path}, pca_dim);
            double identity_rate = eigenface.recognize(face_images_path, eye_locations_path);
            fout << pca_dim << "," << identity_rate << std::endl;
        }
        for (int pca_dim = 100; pca_dim <= 200; pca_dim += 20) {
            std::cout << "pca_dim = " << pca_dim << std::endl;
            EigenFace eigenface({face_images_path, eye_locations_path}, pca_dim);
            double identity_rate = eigenface.recognize(face_images_path, eye_locations_path);
            fout << pca_dim << "," << identity_rate << std::endl;
        }
        fout.close();
    } catch (Exception& e) {
        std::cerr << "Exception: " << e.msg << std::endl;
        return -1;
    }

    return 0;
}