// g++ main.cpp -I /usr/include/opencv4 -L /usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sys/stat.h>

using namespace cv;

#define mat_at(mat, i, j) (((i) < 0 || (i) >= mat.rows || (j) < 0 || (j) >= mat.cols) ? 0 : mat.at<uchar>((i), (j)))

// put here to let the variables be global
// which makes outputting the inner image easier

/*
 * Discrete angle
 * return the sector index of the angle
 * @param 
 * @return: the sector index of the angle
 *         0: -pi/8 ~ pi/8 or -pi ~ -7pi/8 or 7pi/8 ~ pi
 *         1: -7pi/8 ~ -5pi/8 or pi/8 ~ 3pi/8
 *         2: -5pi/8 ~ -3pi/8 or 3pi/8 ~ 5pi/8
 *         3: -3pi/8 ~ -pi/8 or 5pi/8 ~ 7pi/8
 */
uchar giveSector(float angle) {
    // if the angle is negative, add pi to make it positive
    if (angle < 0) {
        angle += CV_PI;
    }

    // get the sector index of the angle
    uchar sector = 0;
    if (angle >= 0 && angle < CV_PI / 8.0) {
        sector = 0;
    } else if (angle >= CV_PI / 8.0 && angle < 3.0 * CV_PI / 8.0) {
        sector = 1;
    } else if (angle >= 3.0 * CV_PI / 8.0 && angle < 5.0 * CV_PI / 8.0) {
        sector = 2;
    } else if (angle >= 5.0 * CV_PI / 8.0 && angle < 7.0 * CV_PI / 8.0) {
        sector = 3;
    } else if (angle >= 7.0 * CV_PI / 8.0 && angle < CV_PI) {
        sector = 0;
    }

    return sector;
}

/*
 * Calculate the gradient magnitude and direction angle
 * @param source: input image
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 */
void calGradandAngle(Mat &source, Mat &grad_m, Mat &dict_angle) {
    float max_grad_m = 0;
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            if (i == source.rows - 1 || j == source.cols - 1) {
                // if the pixel is at the border,
                // set the gradient magnitude to the left pixel's value
                // to avoid abnormal values
                grad_m.at<uchar>(i, j) = grad_m.at<uchar>(i, j-1);
                continue;
            }
            float Gx = (mat_at(source, i, j+1) - mat_at(source, i, j) + mat_at(source, i+1, j+1) - mat_at(source, i+1, j)) / 2.0;
            float Gy = (mat_at(source, i, j) - mat_at(source, i+1, j) + mat_at(source, i, j+1) - mat_at(source, i+1, j+1)) / 2.0;
            
            // calculate gradient magnitude
            grad_m.at<uchar>(i, j) = (uchar)sqrtf32(Gx*Gx + Gy*Gy);
            // update max_grad_m
            if (grad_m.at<uchar>(i, j) > max_grad_m) {
                max_grad_m = grad_m.at<uchar>(i, j);
            }
            // calculate gradient direction angle
            // std::cout << "Gx: " << Gx << " Gy: " << Gy << " atan2: " << atan2(Gy, Gx) << std::endl;
            dict_angle.at<float>(i, j) = atan2f32(Gy, Gx);
        }
    }

    // normalize the gradient magnitude
    for (int i = 0; i < grad_m.rows; i++) {
        for (int j = 0; j < grad_m.cols; j++) {
            grad_m.at<uchar>(i, j) = grad_m.at<uchar>(i, j) / max_grad_m * 255;
        }
    }
}

/*
 * Non-maximum suppression's Implementation.
 * If the pixel is not the local maximum
 * in the direction of the sector, set it to 0.
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 * @param dst: output image
 */
void nonMaxSuppression(Mat &grad_m, Mat &dict_angle, Mat &dst) {
    for (int i = 0; i < grad_m.rows; i++) {
        for (int j = 0; j < grad_m.cols; j++) {
            // get the sector index of the angle
            uchar sector = giveSector(dict_angle.at<float>(i, j));
            // get the gradient magnitude of the pixel
            uchar grad_mag = grad_m.at<uchar>(i, j);
            // get the max gradient magnitude of the pixel
            // in the direction of the sector to compare
            uchar grad_mag_cmp = 0;
            // std::cout << (int)sector << std::endl;
            switch (sector) {
                /* be careful of the direction of the gradient
                 * ----→ Gx
                 *     |
                 *     ↓ Gy
                 * can be viewed as -pi/4, while
                 *     ↑ Gy
                 *     |
                 * ----→ Gx
                 * can be viewed as pi/4
                 */
                case 0: // 0: viewed as 0
                    grad_mag_cmp = max(mat_at(grad_m, i, j-1), mat_at(grad_m, i, j+1));
                    break;
                case 1: // 1: viewed as pi/4
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j-1), mat_at(grad_m, i+1, j+1));
                    break;
                case 2: // 2: viewed as pi/2
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j), mat_at(grad_m, i+1, j));
                    break;
                case 3: // 3: viewed as -pi/4
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j+1), mat_at(grad_m, i+1, j-1));
                    break;
            }

            // if the pixel is not the local maximum, set it to 0
            if (grad_mag < grad_mag_cmp) {
                dst.at<uchar>(i, j) = 0;
            } else {
                dst.at<uchar>(i, j) = grad_mag;
            }
        }
    }
}

/*
 * Hysteresis thresholding's Step 1.
 * Divide into 2 parts:
 * 1. get the strong and weak edges
 * 2. connect the weak edges to the strong edges to get the final edges
 * @param src: input image
 * @param strong_edges: output image (containing strong edges)
 * @param weak_edges: output image (containing weak edges)
 * @param lowThreshold: first threshold for the hysteresis procedure
 * @param highThreshold: second threshold for the hysteresis procedure
 */
void hysteresisThresholdingStep1(Mat &src, Mat &strong_edges, Mat &weak_edges, int lowThreshold, int highThreshold) {
    for (int i = 0; i < src.rows; i++) {
        // std::cout << std::endl << i << " | ";
        for (int j = 0; j < src.cols; j++) {
            // std::cout << j << " ";
            // if the pixel is larger than highThreshold, it is a strong edge
            if (src.at<uchar>(i, j) >= highThreshold) {
                strong_edges.at<uchar>(i, j) = 255;
            } else {
                strong_edges.at<uchar>(i, j) = 0;
            }
            // if the pixel is larger than lowThreshold, it is a weak edge
            if (src.at<uchar>(i, j) >= lowThreshold) {
                weak_edges.at<uchar>(i, j) = 255;
            } else {
                weak_edges.at<uchar>(i, j) = 0;
            }
        }
    }
}

void hysteresisThresholdingStep2(Mat &src, Mat &dst, int lowThreshold, int highThreshold) {
    // Step 2: connect the weak edges to the strong edges to get the final edges
    // std::cout << "connect the weak edges to the strong edges" << std::endl;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // if the pixel is a weak edge
            if (src.at<uchar>(i, j) == 255) {
                // check if it has a strong edge in its 8-neighborhood
                bool has_strong_edge = false;
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        // if the pixel is a strong edge
                        if (dst.at<uchar>(i+k, j+l) == 255) {
                            has_strong_edge = true;
                            break;
                        }
                    }
                    if (has_strong_edge) {
                        break;
                    }
                }
                // if it has a strong edge in its 8-neighborhood, set it to 255
                if (has_strong_edge) {
                    dst.at<uchar>(i, j) = 255;
                } else {
                    dst.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}

/*
 * Hysteresis thresholding's Implementation.
 * Divide into 2 parts:
 * 1. get the strong and weak edges
 * 2. connect the weak edges to the strong edges to get the final edges
 * @param src: input image
 * @param dst: output image
 * @param lowThreshold: first threshold for the hysteresis procedure
 * @param highThreshold: second threshold for the hysteresis procedure
 */
void hysteresisThresholding(Mat &src, OutputArray dst, int lowThreshold, int highThreshold) {
    // Step 1: get the strong and weak edges
    Mat strong_edges, weak_edges;
    strong_edges.create(src.size(), CV_8U);
    weak_edges.create(src.size(), CV_8U);
    hysteresisThresholdingStep1(src, strong_edges, weak_edges, lowThreshold, highThreshold);

    // Step 2: connect the weak edges to the strong edges to get the final edges
    // std::cout << "connect the weak edges to the strong edges" << std::endl;
    hysteresisThresholdingStep2(weak_edges, strong_edges, lowThreshold, highThreshold);

    // output the final edges
    dst.create(src.size(), CV_8U);
    dst.assign(strong_edges);
}

/*
 * Implemented myCanny function
 * (args are the same as Canny function)
 * @param image: input image (gray image)
 * @param edges: output image (containing edges)
 * @param highThreshold: low threshold for the hysteresis procedure
 * @param highThreshold: high threshold for the hysteresis procedure
 * @param apertureSize: aperture size for the Sobel operator
 * @param L2gradient: a flag, indicating whether a more accurate L2 norm
 */
void myCanny(InputArray image, OutputArray edges,
             double lowThreshold, double highThreshold,
             int apertureSize = 3, bool L2gradient = false) { 
    // Step 1: blur the gray image
    // std::cout << "blur" << std::endl;
    Mat blurred;
    blur(image, blurred, Size(3,3));

    // Step 2: calculate gradient magnitude and direction angle
    // std::cout << "grad and angle" << std::endl;
    Mat grad_m, dict_angle;
    // use float to store the gradient magnitude and direction angle
    grad_m.create(blurred.size(), CV_8U);
    dict_angle.create(blurred.size(), CV_32F);
    calGradandAngle(blurred, grad_m, dict_angle);

    // Step 3: non-maximum suppression
    // std::cout << "non-maximum suppression" << std::endl;
    Mat non_max_sup;
    non_max_sup.create(grad_m.size(), CV_8U);
    nonMaxSuppression(grad_m, dict_angle, non_max_sup);

    // Step 4: hysteresis thresholding
    // std::cout << "hysteresis thresholding" << std::endl;
    hysteresisThresholding(non_max_sup, edges, lowThreshold, highThreshold);
}

int main(int argc, char** argv) {
    const String parser_string = "{help h usage    |           | print this message  }"
                                 "{inner           |           | output inner image  }"
                                 "{@input          | lena.jpg  | input image         }"
                                 "{@output         | canny.png | output image        }"
                                 "{t @lowThreshold | 20        | low threshold value }"
                                 "{r @ratio        | 3         | ratio               }"
                                 "{k @kernel_size  | 3         | kernel size         }"
                                 ;
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
    float lowThreshold = parser.get<float>("@lowThreshold");
    float ratio = parser.get<float>("@ratio");
    float kernel_size = parser.get<float>("@kernel_size");

    // create dst image size and get the gray image of src
    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    // detect edges using canny
    myCanny(src_gray, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

    // use detected_edges as a mask to copy the original image
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);

    // output the processed image
    imwrite(output_filename, dst);
    
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
        // output inner image
        /* TODO */
    }
    return 0;
}