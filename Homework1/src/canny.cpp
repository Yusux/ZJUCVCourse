#include "canny.hpp"
#include <iostream>
#include <sys/stat.h>

const String parser_string = "{help h usage    |           | print this message  }"
                             "{inner           |           | output inner image  }"
                             "{@input          | lena.jpg  | input image         }"
                             "{@output         | canny.png | output image        }"
                             "{t lowThreshold  | 20        | low threshold value }"
                             "{r ratio         | 3         | ratio               }"
                             "{k kernel_size   | 3         | kernel size         }"
                             ;

int main(int argc, char** argv) {
    Mat src, src_gray, dst, detected_edges;

    // parse command line arguments
    CommandLineParser parser(argc, argv, parser_string);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // try to open input image
    try {
        src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    } catch (const cv::Exception& e) {
        std::cerr << "\n" << "Error opening file: " << e.what() << std::endl;
        parser.printMessage();
        return -1;
    }

    // initialize variables
    bool output_inner = parser.has("inner");
    String output_filename = parser.get<String>("@output");
    float lowThreshold = parser.get<float>("lowThreshold");
    float ratio = parser.get<float>("ratio");
    float kernel_size = parser.get<float>("kernel_size");
    
    // if output_inner is true, output the inner image
    if (output_inner) {
        // mkdir inner
        String inner_dir = "inner";
        // check if inner directory exists
        struct stat info;
        if (stat(inner_dir.c_str(), &info) != 0) {
            // create inner directory
            mkdir(inner_dir.c_str(), 0755);
        }
    }

    // create dst image size and get the gray image of src
    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    // detect edges using canny
    myCanny(src_gray, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size, output_inner);

    // use detected_edges as a mask to copy the original image
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);

    // output the processed image
    imwrite(output_filename, dst);

    return 0;
}