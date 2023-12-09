#include "utils.hpp"
#include <sys/stat.h>

using namespace cv;

void checkFolder(String input_dir, std::vector<String> &filenames, bool recursive) {
    // check if input directory exists
    struct stat info;
    if (stat(input_dir.c_str(), &info) != 0) {
        Exception e;
        e.msg = "Input directory `" + input_dir + "` does not exist.";
        throw e;
    }
    glob(input_dir, filenames, recursive);
}

void checkFile(String input_file) {
    // check if input file exists
    struct stat info;
    if (stat(input_file.c_str(), &info) != 0) {
        Exception e;
        e.msg = "Input file `" + input_file + "` does not exist.";
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

void getMaxInnerRect(Mat &mask, Rect &rect) {
    // get the min enclosing rect
    getMinEnclosingRect(mask, rect);
    // assert rect is larger than 2x2
    assert(rect.width > 2 && rect.height > 2);
    // smallen the rect by 1 pixel
    rect.x += 1;
    rect.y += 1;
    rect.width -= 2;
    rect.height -= 2;
    // get the initial rect of the min enclosing rect
    Mat min_rect_mask = Mat::zeros(mask.size(), CV_8UC1);
    // draw the initial rect
    rectangle(min_rect_mask, rect, Scalar(255), FILLED);
    // get the sub mask
    Mat sub_mask = min_rect_mask - mask;

    // while there is still non-zero pixel in the sub mask
    // which means there is still black pixel in the rect
    while (countNonZero(sub_mask) > 0) {
        // erode the min rect mask
        erode(min_rect_mask, min_rect_mask, Mat());
        // get the sub mask
        sub_mask = min_rect_mask - mask;
    }

    // Assert the min rect mask is not empty
    assert(countNonZero(min_rect_mask) > 0);

    // get the max inner rect
    getMinEnclosingRect(min_rect_mask, rect);
}

Point2i getRectCenter(Rect rect) {
    return Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2);
}
