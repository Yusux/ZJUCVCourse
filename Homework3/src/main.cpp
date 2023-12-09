#include "eigenface.hpp"
#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <ctime>

using namespace cv;

const String parser_string = "{help h usage     |                   | print this message                                }"
                             "{config_path c    | eigenface.yml     | config file path                                  }"
                             "{inner            |                   | output inner image                                }"
                             "{work_type t      | train             | work type, can be train, recognize or reconstruct }"
                             "{energy_ratio r   | 0.99              | energy ratio for eigenfaces                       }"
                             "{eye_loctation e  |                   | eye location json file or directory path          }"
                             "{@input           | att/att-face      | input images directory or config file path        }"
                             "{@output          | reconstructed.png | output image name                                 }"
                             ;

int main(int argc, char** argv) {
    // parse command line arguments
    CommandLineParser parser(argc, argv, parser_string);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // get the parsed arguments
    bool output_inner = parser.has("inner");
    String config_path = parser.get<String>("config_path");
    String work_type = parser.get<String>("work_type");
    double energy_ratio = parser.get<double>("energy_ratio");
    String eye_location_path;
    if (parser.has("eye_loctation")) {
        eye_location_path = parser.get<String>("eye_loctation");
    }
    String input_path = parser.get<String>("@input");
    String output_filename = parser.get<String>("@output");

    std::cout << "# Arguments #" << std::endl;
    std::cout << std::setw(20) << std::left << "config_path: "       << std::left << config_path << std::endl;
    std::cout << std::setw(20) << std::left << "output_inner: "      << std::left << output_inner << std::endl;
    std::cout << std::setw(20) << std::left << "work_type: "         << std::left << work_type << std::endl;
    std::cout << std::setw(20) << std::left << "energy_ratio: "      << std::left << energy_ratio << std::endl;
    std::cout << std::setw(20) << std::left << "eye_location_path: " << std::left << eye_location_path << std::endl;
    std::cout << std::setw(20) << std::left << "input_path: "        << std::left << input_path << std::endl;
    std::cout << std::setw(20) << std::left << "output_filename: "   << std::left << output_filename << std::endl;
    std::cout << "# End of arguments #" << std::endl << std::endl;
    
    // if output_inner is true, output the inner image
    if (output_inner) {
        // mkdir inner
        createFolder("inner");
    }

    clock_t start, end;
    start = clock();
    try {
        if (work_type == "train") {
            EigenFace eigenface(true, {config_path, input_path, eye_location_path}, output_inner, energy_ratio);
        } else if (work_type == "recognize") {
            EigenFace eigenface(false, {config_path}, output_inner);
            String identity = eigenface.recognize(input_path, eye_location_path);
            std::cout << "The identity of the face is " << identity << std::endl;
        } else if (work_type == "reconstruct") {
            EigenFace eigenface(false, {config_path}, output_inner);
            Mat dst = eigenface.reconstruct(input_path, eye_location_path);
            imwrite(output_filename, dst);
        } else {
            Exception e;
            e.msg = "Work type `" + work_type + "` is not supported.";
            throw e;
        }
    } catch (Exception& e) {
        std::cerr << "Exception: " << e.msg << std::endl;
        return -1;
    }
    end = clock();
    std::cout << work_type << " ends (elapsed time: " << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;

    return 0;
}