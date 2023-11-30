#include "stitching.hpp"
#include "blenders.hpp"
#include "utils.hpp"
#include <iostream>

using namespace cv;

void MyStitcher::stitch() {
    std::cout << "Stitching based on " << filenames_[0] << std::endl;
    for (int i = 1; i < filenames_.size(); i++) {
        std::cout << "Stitching " << filenames_[i] << " to the base image" << std::endl;
        // Step 2: Feature Matching
        // **choose the best match image**
        // prepare the srcB from the dst
        Mat srcA, srcB;
        std::vector<KeyPoint> keypointsA, keypointsB;
        std::vector<DMatch> best_matches;
        featureMatching(srcA, srcB, keypointsA, keypointsB, best_matches);

        // output the inner image
        if (output_inner_) {
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
        Mat H;
        try {
            homographyEstimation(srcA, srcB, keypointsA, keypointsB, best_matches, H);
        } catch (const Exception& e) {
            throw e;
        }

        // Step 4: Image Warping
        Mat base, warpedA;
        imageWarping(srcA, srcB, H, base, warpedA);
        if (output_inner_) {
            imwrite("inner/" + std::to_string(i-1) + "_base.png", base);
            imwrite("inner/" + std::to_string(i-1) + "_warpedA.png", warpedA);
        }

        // Step 5: Image Blending
        imageBlending(base, warpedA);

        // output the inner image
        if (output_inner_) {
            imwrite("inner/" + std::to_string(i-1) + "_stitiched.png", dst_);
        }
    }
}

MyStitcher::MyStitcher(std::vector<cv::String> filenames,
                       int blend_type,
                       bool output_inner) {
    CV_Assert(filenames.size() >= 2);
    CV_Assert(blend_type >= 0 && blend_type <= 4);

    output_inner_ = output_inner;
    blend_type_ = blend_type;
    filenames_ = filenames;

    // initialize sift
    sift_ = SIFT::create();

    try {
        // prepare the images
        prepareImages();
        // detect the features
        detectFeatures();
    } catch (const Exception& e) {
        throw e;
    }
}

MyStitcher::~MyStitcher() {
}

Mat MyStitcher::getStitchedImg() {
    return dst_;
}

void MyStitcher::prepareImages() {
    // Step 0: Prepare the images
    // convert the images to grayscale
    // read the all the images
    // check if the images are read successfully
    // and convert the images to grayscale
    for (int i = 1; i < filenames_.size(); i++) {
        Mat img = imread(filenames_[i]);
        checkImg(img.data, filenames_[i]);
        images_.push_back(img);
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        images_gray_.push_back(img_gray);
    }
    dst_ = imread(filenames_[0]);
    checkImg(dst_.data, filenames_[0]);
}

void MyStitcher::detectFeatures() {
    // Step 1: Feature Detection
    // use SIFT to detect keypoints and extract descriptors
    for (int i = 0; i < filenames_.size()-1; i++) {
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_->detectAndCompute(images_gray_[i], Mat(), keypoints, descriptors);
        images_keypoints_.push_back(keypoints);
        images_descriptors_.push_back(descriptors);
    }
}

void MyStitcher::featureMatching(Mat &srcA, Mat &srcB, 
                                 std::vector<KeyPoint> &keypointsA,
                                 std::vector<KeyPoint> &keypointsB,
                                 std::vector<DMatch> &best_matches) {
    // Step 2: Feature Matching
    // **choose the best match image**
    // prepare the srcB from the dst
    srcB = dst_;
    Mat descriptorsB;
    sift_->detectAndCompute(srcB, Mat(), keypointsB, descriptorsB);

    // use BFMatcher to match the descriptors
    // the knnmatch is not the knn used to classify
    // it is the k nearest neighbors (in the trainImage)
    // of each descriptor in the queryImage
    // refer to https://docs.opencv.org/4.x/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
    BFMatcher matcher(NORM_L2, false);
    int idxA = 0;
    for (int j = 0; j < images_descriptors_.size(); j++) {
        std::vector<std::vector<DMatch>> matches;
        matcher.knnMatch(images_descriptors_[j], descriptorsB, matches, 2);
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
    srcA = images_[idxA];
    keypointsA = images_keypoints_[idxA];

    // pop the idxA image
    images_.erase(images_.begin() + idxA);
    images_gray_.erase(images_gray_.begin() + idxA);
    images_keypoints_.erase(images_keypoints_.begin() + idxA);
    images_descriptors_.erase(images_descriptors_.begin() + idxA);
}

void MyStitcher::homographyEstimation(Mat &srcA, Mat &srcB,
                                      std::vector<KeyPoint> &keypointsA,
                                      std::vector<KeyPoint> &keypointsB,
                                      std::vector<DMatch> &best_matches,
                                      Mat &H) {
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
    H = findHomography(srcA_pts, srcB_pts, RANSAC, 5.0);
}

void MyStitcher::imageWarping(Mat &srcA, Mat &srcB, Mat &H,
                              Mat &base, Mat &warped) {
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
    projected_bounds[0] = floorMin(projected_corners[0].y, projected_corners[1].y,
                                   projected_corners[2].y, projected_corners[3].y); // top
    projected_bounds[1] = ceilMax(projected_corners[0].y, projected_corners[1].y,
                                  projected_corners[2].y, projected_corners[3].y);  // bottom
    projected_bounds[2] = floorMin(projected_corners[0].x, projected_corners[1].x,
                                   projected_corners[2].x, projected_corners[3].x); // left
    projected_bounds[3] = ceilMax(projected_corners[0].x, projected_corners[1].x,
                                  projected_corners[2].x, projected_corners[3].x);  // right
    std::vector<int> padding_size(4);
    padding_size[0] = std::max(0, -projected_bounds[0]);                            // top
    padding_size[1] = std::max(0, projected_bounds[1] - srcB.rows);                 // bottom
    padding_size[2] = std::max(0, -projected_bounds[2]);                            // left
    padding_size[3] = std::max(0, projected_bounds[3] - srcB.cols);                 // right
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
    warpPerspective(srcA, warped, H, Size(canvas_width, canvas_height), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    
    // Step 4.3: Copy the images to the canvas
    // create the canvas
    base = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    // copy the base image to the dst image with the mask
    srcB.copyTo(Mat(base, Rect(padding_size[2], padding_size[0], srcB.cols, srcB.rows)));
}

void MyStitcher::imageBlending(Mat &base, Mat &warped) {
    CV_Assert(base.size() == warped.size() && base.type() == warped.type());
    int canvas_width = base.cols;
    int canvas_height = base.rows;

    CV_Assert(blend_type_ >= 0 && blend_type_ <= 4);
    if (blend_type_ == NO_BLEND) {
        // 0. not use blending
        // copy the warped image to the dst image with the mask
        // create mask for the warped image
        Mat maskA;
        getMask(warped, maskA, 1, 2);
        dst_ = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
        base.copyTo(Mat(dst_, Rect(0, 0, base.cols, base.rows)));
        warped.copyTo(Mat(dst_, Rect(0, 0, warped.cols, warped.rows)), maskA);
    } else if (blend_type_ == LINEAR_BLEND) {
        // 1. use linear blend to blend the base and warped image
        linearBlend(base, warped, dst_);
    } else if (blend_type_ == MY_MULTIBAND_BLEND) {
        // 2. use self implemented multi-band blending to blend the base and warped image
        std::vector<Mat> images;
        images.push_back(base);
        images.push_back(warped);
        MyMultiBandBlender blender(images);
        blender.blend(dst_);
    } else if (blend_type_ == OPENCV_MULTIBAND_BLEND) {
        // 3. use opencv multi-band blending to blend the base and warped image
        Mat base_mask, warped_mask;
        getMask(base, base_mask);
        getMask(warped, warped_mask, 1, 2);
        Rect base_rect, warped_rect;
        getMinEnclosingRect(base_mask, base_rect);
        getMinEnclosingRect(warped_mask, warped_rect);
        // create the blender
        detail::MultiBandBlender blender(false);
        blender.prepare(Rect(0, 0, canvas_width, canvas_height));
        blender.feed(base, base_mask, Point(0, 0));
        blender.feed(warped, warped_mask, Point(0, 0));
        blender.blend(dst_, Mat());
        dst_.convertTo(dst_, CV_8UC3);
    } else{
        // 4. use alpha blending to blend the base and warped image
        // refer to https://stackoverflow.com/a/22324790
        Mat maskA, maskB;
        getMask(warped, maskA, 1, 2);
        getMask(base, maskB);
        dst_ = computeAlphaBlending(warped, maskA, base, maskB);
    }
}