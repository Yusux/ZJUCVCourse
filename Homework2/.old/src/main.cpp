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

int floorMin(float a, float b, float c, float d) {
    int floor_a = (int)floor(a);
    int floor_b = (int)floor(b);
    int floor_c = (int)floor(c);
    int floor_d = (int)floor(d);
    return std::min(std::min(std::min(floor_a, floor_b), floor_c), floor_d);
}

int ceilMax(float a, float b, float c, float d) {
    int ceil_a = (int)ceil(a);
    int ceil_b = (int)ceil(b);
    int ceil_c = (int)ceil(c);
    int ceil_d = (int)ceil(d);
    return std::max(std::max(std::max(ceil_a, ceil_b), ceil_c), ceil_d);
}

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
        // Step 0: Prepare the images
        // convert the images to grayscale
        // read the all the images
        // check if the images are read successfully
        // and convert the images to grayscale
        std::vector<Mat> images;
        std::vector<Mat> images_gray;
        for (int i = 0; i < filenames.size()-1; i++) {
            Mat img = imread(filenames[i]);
            checkImg(img.data, filenames[i]);
            images.push_back(img);
            Mat img_gray;
            cvtColor(img, img_gray, COLOR_BGR2GRAY);
            images_gray.push_back(img_gray);
        }
        dst = imread(filenames[filenames.size()-1]);
        checkImg(dst.data, filenames[filenames.size()-1]);

        // Step 1: Feature Detection
        // use SIFT to detect keypoints and extract descriptors
        Ptr<SIFT> sift = SIFT::create();
        std::vector<std::vector<KeyPoint>> images_keypoints;
        std::vector<Mat> images_descriptors;
        for (int i = 0; i < filenames.size()-1; i++) {
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            sift->detectAndCompute(images_gray[i], Mat(), keypoints, descriptors);
            images_keypoints.push_back(keypoints);
            images_descriptors.push_back(descriptors);
        }

        for (int i = 1; i < filenames.size(); i++) {
            // Step 2: Feature Matching
            // **choose the best match image**
            // prepare the srcB from the dst
            Mat srcB = dst;
            std::vector<KeyPoint> keypointsB;
            Mat descriptorsB;
            sift->detectAndCompute(srcB, Mat(), keypointsB, descriptorsB);

            // use BFMatcher to match the descriptors
            // the knnmatch is not the knn used to classify
            // it is the k nearest neighbors (in the trainImage)
            // of each descriptor in the queryImage
            // refer to https://docs.opencv.org/4.x/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
            BFMatcher matcher(NORM_L2, false);
            std::vector<DMatch> best_matches;
            int idxA = 0;
            for (int j = 0; j < images_descriptors.size(); j++) {
                std::vector<std::vector<DMatch>> matches;
                matcher.knnMatch(images_descriptors[j], descriptorsB, matches, 2);
                // add ratio test, refer to Figure 11 in paper
                // Distinctive Image Features from Scale-Invariant Keypoints
                // > In other documents (https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html),
                // > it is also mentioned that
                // > (BFMatcher) Second param is boolean variable,
                // > crossCheck which is false by default. If it is true
                // > ... It provides consistent result, and is a good alternative
                // > to ratio test proposed by D.Lowe in SIFT paper.
                std::vector<DMatch> good_matches;
                for (int k = 0; k < matches.size(); k++) {
                    if (matches[k][0].distance < 0.75 * matches[k][1].distance) {
                        good_matches.push_back(matches[k][0]);
                    }
                }
                if (good_matches.size() > best_matches.size()) {
                    best_matches = good_matches;
                    idxA = j;
                }
            }

            // prepare the srcA from the best match image
            Mat &srcA = images[idxA];
            std::vector<KeyPoint> &keypointsA = images_keypoints[idxA];
            Mat &descriptorsA = images_descriptors[idxA];

            // output the inner image
            if (output_inner) {
                Mat inner_A, inner_B;
                drawKeypoints(srcA, keypointsA, inner_A);
                drawKeypoints(srcB, keypointsB, inner_B);
                imwrite("inner/" + std::to_string(i-1) + "_keypoints_A.png", inner_A);
                imwrite("inner/" + std::to_string(i-1) + "_keypoints_B.png", inner_B);
                Mat inner;
                drawMatches(srcA, keypointsA, srcB, keypointsB, best_matches, inner);
                imwrite("inner/" + std::to_string(i-1) + "_matches.png", inner);
            }

            // Step 3: Homography Estimation
            // use RANSAC to estimate the homography matrix
            if (best_matches.size() < 4) {
                Exception e;
                e.msg = "Not enough matches.";
                throw e;
            }
            std::vector<Point2f> srcA_pts, srcB_pts;
            for (int j = 0; j < best_matches.size(); j++) {
                srcA_pts.push_back(keypointsA[best_matches[j].queryIdx].pt);
                srcB_pts.push_back(keypointsB[best_matches[j].trainIdx].pt);
            }
            Mat H = findHomography(srcA_pts, srcB_pts, RANSAC, 5.0);

            // Step 4: Image Warping
            // Step 4.1: Calculate the size of the canvas
            // refer to https://stackoverflow.com/questions/31978721/opencv-image-stitching-leaves-a-blank-region-after-the-right-boundary
            std::vector<Point2f> srcA_corners(4);
            srcA_corners[0] = Point2f(0, 0);
            srcA_corners[1] = Point2f(srcA.cols, 0);
            srcA_corners[2] = Point2f(srcA.cols, srcA.rows);
            srcA_corners[3] = Point2f(0, srcA.rows);
            std::vector<Point2f> projected_corners(4);
            perspectiveTransform(srcA_corners, projected_corners, H);
            // get the final padding size
            std::vector<int> projected_bounds(4);
            projected_bounds[0] = floorMin(projected_corners[0].y, projected_corners[1].y, projected_corners[2].y, projected_corners[3].y); // top
            projected_bounds[1] = ceilMax(projected_corners[0].y, projected_corners[1].y, projected_corners[2].y, projected_corners[3].y);  // bottom
            projected_bounds[2] = floorMin(projected_corners[0].x, projected_corners[1].x, projected_corners[2].x, projected_corners[3].x); // left
            projected_bounds[3] = ceilMax(projected_corners[0].x, projected_corners[1].x, projected_corners[2].x, projected_corners[3].x);  // right
            std::vector<int> padding_size(4);
            padding_size[0] = std::max(0, -projected_bounds[0]);                                                    // top
            padding_size[1] = std::max(0, projected_bounds[1] - srcB.rows);                                     // bottom
            padding_size[2] = std::max(0, -projected_bounds[2]);                                                    // left
            padding_size[3] = std::max(0, projected_bounds[3] - srcB.cols);                                     // right
            // get the final canvas size
            int canvas_width = srcB.cols + padding_size[2] + padding_size[3];
            int canvas_height = srcB.rows + padding_size[0] + padding_size[1];

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
            Mat warpedA;
            warpPerspective(srcA, warpedA, H, Size(canvas_width, canvas_height));
            if (output_inner) {
                imwrite("inner/" + std::to_string(i-1) + "_warp_pre.png", warpedA);
            }
            // create mask for the warped image
            Mat maskA = Mat::zeros(warpedA.rows, warpedA.cols, CV_8UC1);
            for (int j = 0; j < warpedA.rows; j++) {
                for (int k = 0; k < warpedA.cols; k++) {
                    if (warpedA.at<Vec3b>(j, k) != Vec3b(0, 0, 0)) {
                        maskA.at<uchar>(j, k) = 255;
                    }
                }
            }
            // erode the mask for 1 pixel
            erode(maskA, maskA, Mat(), Point(-1, -1), 1);
            
            // Step 4.3: Copy the images to the canvas
            // create the canvas
            dst = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
            // copy the base image to the dst image with the mask
            srcB.copyTo(Mat(dst, Rect(padding_size[2], padding_size[0], srcB.cols, srcB.rows)));
            // copy the warped image to the dst image with the mask
            warpedA.copyTo(Mat(dst, Rect(0, 0, warpedA.cols, warpedA.rows)), maskA);
            // output the inner image
            if (output_inner) {
                imwrite("inner/" + std::to_string(i-1) + "_warp.png", dst);
            }

            // pop the idxA image
            images.erase(images.begin() + idxA);
            images_gray.erase(images_gray.begin() + idxA);
            images_keypoints.erase(images_keypoints.begin() + idxA);
            images_descriptors.erase(images_descriptors.begin() + idxA);
        }
    } catch (const Exception& e) {
        std::cerr << "Error: " << e.msg << std::endl;
        return -1;
    }

    // output the processed image
    imwrite(output_filename, dst);

    return 0;
}