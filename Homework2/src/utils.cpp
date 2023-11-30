#include "utils.hpp"
#include <sys/stat.h>

using namespace cv;

void checkFolder(String input_dir, std::vector<String> &filenames) {
    // check if input directory exists
    struct stat info;
    if (stat(input_dir.c_str(), &info) != 0) {
        Exception e;
        e.msg = "Input directory `" + input_dir + "` does not exist.";
        throw e;
    }
    glob(input_dir, filenames, false);
    if (filenames.size() < 2) {
        Exception e;
        e.msg = "Input directory `" + input_dir + "` does not contain enough images.";
        throw e;
    }
}

void createFolder(String output_dir) {
    // check if output directory exists
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0) {
        // create output directory
        mkdir(output_dir.c_str(), 0755);
    }
}

void checkImg(uchar* data, String filename) {
    if (data == NULL) {
        Exception e;
        e.msg = "Cannot read image `" + filename + "`.";
        throw e;
    }
}

int floorMin(float a, float b, float c, float d) {
    int floor_a = (int)(floor(a));
    int floor_b = (int)(floor(b));
    int floor_c = (int)(floor(c));
    int floor_d = (int)(floor(d));
    return std::min(std::min(std::min(floor_a, floor_b), floor_c), floor_d);
}

int ceilMax(float a, float b, float c, float d) {
    int ceil_a = (int)(ceil(a));
    int ceil_b = (int)(ceil(b));
    int ceil_c = (int)(ceil(c));
    int ceil_d = (int)(ceil(d));
    return std::max(std::max(std::max(ceil_a, ceil_b), ceil_c), ceil_d);
}

void getMask(Mat &img, Mat &mask, int dilate_size, int erode_size) {
    // create mask for the image
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    threshold(img_gray, mask, 0, 255, THRESH_BINARY);

    // dilate and erode the mask
    if (dilate_size != -1) {
        dilate(mask, mask, Mat(), Point(-1, -1), dilate_size);
    }
    if (erode_size != -1) {
        erode(mask, mask, Mat(), Point(-1, -1), erode_size);
    }
}

void getMinEnclosingRect(Mat &mask, Rect &rect) {
    // use boundingRect to get the min enclosing rect
    rect = boundingRect(mask);
}