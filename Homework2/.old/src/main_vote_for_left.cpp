#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <sys/stat.h>

using namespace cv;

const String parser_string = "{help h usage    |              | print this message     }"
                             "{inner           |              | output inner image     }"
                             "{@input          | pictures     | input images directory }"
                             "{@output         | stitched.png | output image           }"
                             ;

void checkImg(uchar* data, String filename) {
    if (data == NULL) {
        Exception e;
        e.msg = "Cannot read image `" + filename + "`.";
        throw e;
    }
}

int main(int argc, char** argv) {

    // initialize variables
    String input_dir;
    std::vector<String> filenames;
    Mat srcA, srcA_gray,
        srcB, srcB_gray,
        dst;

    // parse command line arguments
    CommandLineParser parser(argc, argv, parser_string);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // check the input images directory
    // count the number of images in the directory
    try {
        String input_dir = parser.get<String>("@input");
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
    } catch (const Exception& e) {
        std::cerr << "Error: " << e.msg << std::endl;
        parser.printMessage();
        return -1;
    }

    // get the parsed arguments
    bool output_inner = parser.has("inner");
    String output_filename = parser.get<String>("@output");
    
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

    // process each pair of images
    try {
        // read the first image
        dst = imread(filenames[0]);
        checkImg(dst.data, filenames[0]);
        for (int i = 1; i < filenames.size(); i++) {
            // read the images
            srcA = dst;
            srcB = imread(filenames[i]);
            checkImg(srcB.data, filenames[i]);

            // Step 0: Convert the images to grayscale
            cvtColor(srcA, srcA_gray, COLOR_BGR2GRAY);
            cvtColor(srcB, srcB_gray, COLOR_BGR2GRAY);

            // Step 1: Feature Detection
            // use SIFT to detect keypoints and extract descriptors
            Ptr<SIFT> sift = SIFT::create();
            std::vector<KeyPoint> keypointsA, keypointsB;
            Mat descriptorsA, descriptorsB;
            sift->detectAndCompute(srcA_gray, noArray(), keypointsA, descriptorsA, false);
            sift->detectAndCompute(srcB_gray, noArray(), keypointsB, descriptorsB, false);
            // output the inner image
            if (output_inner) {
                Mat innerA, innerB;
                drawKeypoints(srcA, keypointsA, innerA);
                drawKeypoints(srcB, keypointsB, innerB);
                imwrite("inner/" + std::to_string(i-1) + "_A.png", innerA);
                imwrite("inner/" + std::to_string(i-1) + "_B.png", innerB);
            }

            // Step 2: Feature Matching
            // use BFMatcher to match the descriptors
            // the knnmatch is not the knn used to classify
            // it is the k nearest neighbors (in the trainImage)
            // of each descriptor in the queryImage
            // refer to https://docs.opencv.org/4.x/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
            BFMatcher matcher(NORM_L2, false);
            std::vector<std::vector<DMatch>> matches;
            matcher.knnMatch(descriptorsA, descriptorsB, matches, 2);
            // add ratio test, refer to Figure 11 in paper
            // Distinctive Image Features from Scale-Invariant Keypoints
            // > In other documents (https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html),
            // > it is also mentioned that
            // > (BFMatcher) Second param is boolean variable,
            // > crossCheck which is false by default. If it is true
            // > ... It provides consistent result, and is a good alternative
            // > to ratio test proposed by D.Lowe in SIFT paper.
            std::vector<DMatch> good_matches;
            for (int j = 0; j < matches.size(); j++) {
                if (matches[j][0].distance < 0.75 * matches[j][1].distance) {
                    good_matches.push_back(matches[j][0]);
                }
            }
            // judge if srcA is the left image
            // if not, swap srcA and srcB
            // vote < 0: srcA is the left image
            int vote = 0;
            for (int j = 0; j < good_matches.size(); j++) {
                if (keypointsA[good_matches[j].queryIdx].pt.x < keypointsB[good_matches[j].trainIdx].pt.x) {
                    vote++;
                } else {
                    vote--;
                }
            }

            // output the inner image
            if (output_inner) {
                Mat inner;
                drawMatches(srcA, keypointsA, srcB, keypointsB, good_matches, inner);
                imwrite("inner/" + std::to_string(i-1) + "_matches.png", inner);
            }

            // Step 3: Homography Estimation
            // use RANSAC to estimate the homography matrix
            if (good_matches.size() < 4) {
                Exception e;
                e.msg = "Not enough matches.";
                throw e;
            }
            std::vector<Point2f> srcA_pts, srcB_pts;
            for (int j = 0; j < good_matches.size(); j++) {
                srcA_pts.push_back(keypointsA[good_matches[j].queryIdx].pt);
                srcB_pts.push_back(keypointsB[good_matches[j].trainIdx].pt);
            }
            std::cout << "vote: " << vote << std::endl;
            Mat H, *to_warp, *to_base;
            if (vote < 0) {
                H = findHomography(srcB_pts, srcA_pts, RANSAC);
                to_warp = &srcB;
                to_base = &srcA;
            } else {
                H = findHomography(srcA_pts, srcB_pts, RANSAC);
                to_warp = &srcA;
                to_base = &srcB;
            }

            // Step 4: Image Warping
            // Step 4.1: Calculate the size of the canvas
            // refer to https://stackoverflow.com/questions/31978721/opencv-image-stitching-leaves-a-blank-region-after-the-right-boundary
            std::vector<Point2f> to_warp_corners(4);
            to_warp_corners[0] = Point2f(0, 0);
            to_warp_corners[1] = Point2f(to_warp->cols, 0);
            to_warp_corners[2] = Point2f(to_warp->cols, to_warp->rows);
            to_warp_corners[3] = Point2f(0, to_warp->rows);
            std::vector<Point2f> projected_corners(4);
            perspectiveTransform(to_warp_corners, projected_corners, H);
            std::cout << "projected_corners: " << std::endl;
            for (int j = 0; j < 4; j++) {
                std::cout << projected_corners[j] << std::endl;
            }
            // get the final padding size
            std::vector<int> projected_bounds(4);
            projected_bounds[0] = std::min((int)floor(projected_corners[0].y), (int)floor(projected_corners[1].y)); // top
            projected_bounds[1] = std::max((int)ceil(projected_corners[2].y), (int)ceil(projected_corners[3].y));   // bottom
            projected_bounds[2] = std::min((int)floor(projected_corners[0].x), (int)floor(projected_corners[3].x)); // left
            projected_bounds[3] = std::max((int)ceil(projected_corners[1].x), (int)ceil(projected_corners[2].x));   // right
            std::vector<int> padding_size(4);
            padding_size[0] = std::max(0, -projected_bounds[0]);                                                    // top
            padding_size[1] = std::max(0, projected_bounds[1] - to_base->rows);                                     // bottom
            padding_size[2] = std::max(0, -projected_bounds[2]);                                                    // left
            padding_size[3] = std::max(0, projected_bounds[3] - to_base->cols);                                     // right
            for (int j = 0; j < 4; j++) {
                std::cout << "padding_size[" << j << "]: " << padding_size[j] << std::endl;
            }
            int canvas_width = to_base->cols + padding_size[2] + padding_size[3];
            int canvas_height = to_base->rows + padding_size[0] + padding_size[1];

            // Step 4.2: Warp the images
            // apply Ht as the transformation for moving the image
            // refer to https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
            /*
             * [1 0 tx]
             * [0 1 ty]
             * [0 0 1 ]
             * where tx = padding_size[2], ty = padding_size[0]
             */
            Mat Ht = Mat::eye(3, 3, CV_64F);
            Ht.at<double>(0, 2) = padding_size[2];
            Ht.at<double>(1, 2) = padding_size[0];
            H = Ht * H;
            // use the homography matrix to warp the images
            warpPerspective(*to_warp, dst, H, Size(canvas_width, canvas_height));
            if (output_inner) {
                imwrite("inner/" + std::to_string(i-1) + "_warp_pre.png", dst);
            }
            // copy the base image to the dst image
            to_base->copyTo(Mat(dst, Rect(padding_size[2], padding_size[0], to_base->cols, to_base->rows)));
            // output the inner image
            if (output_inner) {
                imwrite("inner/" + std::to_string(i-1) + "_warp.png", dst);
            }
        }
    } catch (const Exception& e) {
        std::cerr << "Error: " << e.msg << std::endl;
        return -1;
    }

    // // output the processed image
    // imwrite(output_filename, dst);

    return 0;
}