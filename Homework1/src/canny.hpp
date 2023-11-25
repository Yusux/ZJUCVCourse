#ifndef __CANNY_HPP__
#define __CANNY_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

#define mat_at(mat, i, j) (((i) < 0 || (i) >= (mat).rows || (j) < 0 || (j) >= (mat).cols) ? 0 : (mat).at<uchar>((i), (j)))

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
uchar giveSector(float angle);

/*
 * Calculate the gradient magnitude and direction angle
 * @param source: input image
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 */
void calGradandAngle(Mat &source, Mat &grad_m, Mat &dict_angle);

/*
 * Non-maximum suppression's Implementation.
 * If the pixel is not the local maximum
 * in the direction of the sector, set it to 0.
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 * @param dst: output image
 */
void nonMaxSuppression(Mat &grad_m, Mat &dict_angle, Mat &dst);

/*
 * Hysteresis thresholding's Step 1
 * Get the strong and weak edges
 * @param src: input image
 * @param strong_edges: output image (containing strong edges)
 * @param weak_edges: output image (containing weak edges)
 * @param lowThreshold: first threshold for the hysteresis procedure
 * @param highThreshold: second threshold for the hysteresis procedure
 */
void hysteresisThresholdingStep1(Mat &src,
                                 Mat &strong_edges, Mat &weak_edges,
                                 int lowThreshold, int highThreshold);
/*
 * Hysteresis thresholding's Step 2
 * Connect the weak edges to the strong edges to get the final edges
 * @param src: input image (containing weak edges)
 * @param dst: output image (containing strong edges)
 */
void hysteresisThresholdingStep2(Mat &src, Mat &dst);

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
void hysteresisThresholding(Mat &src, OutputArray dst, int lowThreshold, int highThreshold);

/*
 * Implemented myCanny function
 * (args are similar to Canny function)
 * @param image: input image (gray image)
 * @param edges: output image (containing edges)
 * @param highThreshold: low threshold for the hysteresis procedure
 * @param highThreshold: high threshold for the hysteresis procedure
 * @param apertureSize: aperture size for the Sobel operator, NOT USED
 * @param outputInner: a flag, indicating whether output the inner image
 * @param L2gradient: a flag, indicating whether a more accurate L2 norm
 */
void myCanny(InputArray image, OutputArray edges,
             double lowThreshold, double highThreshold,
             int apertureSize = 3, bool outputInner = false,
              bool L2gradient = false);

#endif
