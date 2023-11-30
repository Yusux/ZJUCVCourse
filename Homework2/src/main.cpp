#include "utils.hpp"
#include "blenders.hpp"
#include "stitching.hpp"
#include <iostream>

using namespace cv;

const String parser_string = "{help h usage    |              | print this message      }"
                             "{inner           |              | output inner image      }"
                             "{blend_type bt   | 1            | blend type, from 0 to 4, each number represents a blend type:\n\t\t0 for no blend, 1 for linear blend, 2 for my multiband blend,\n\t\t3 for opencv multiband blend, 4 for alpha blend}"
                             "{@input          | pictures     | input images directory  }"
                             "{@output         | stitched.png | output image            }"
                             ;

int main(int argc, char** argv) {

    // initialize variables
    String input_dir;
    std::vector<String> filenames;
    Mat dst;

    // parse command line arguments
    CommandLineParser parser(argc, argv, parser_string);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // check the input images directory
    // count the number of images in the directory
    try {
        input_dir = parser.get<String>("@input");
        checkFolder(input_dir, filenames);
    } catch (const Exception& e) {
        std::cerr << "Error: " << e.msg << std::endl;
        parser.printMessage();
        return -1;
    }

    // get the parsed arguments
    bool output_inner = parser.has("inner");
    int blend_type = parser.get<int>("blend_type");
    String output_filename = parser.get<String>("@output");
    
    // if output_inner is true, output the inner image
    if (output_inner) {
        // mkdir inner
        createFolder("inner");
    }

    // process each pair of images
    try {
        MyStitcher stitcher(filenames, blend_type, output_inner);
        stitcher.stitch();
        dst = stitcher.getStitchedImg();
    } catch (const Exception& e) {
        std::cerr << "Error: " << e.msg << std::endl;
        return -1;
    }

    // output the processed image
    imwrite(output_filename, dst);

    return 0;
}