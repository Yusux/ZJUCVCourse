#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

/*
 * Check if the input directory exists,
 * all files in the directory can be read as images,
 * and contains enough images.
 * @param input_dir: the input directory
 * @param filenames: the filenames of the images
 */
void checkFolder(cv::String input_dir, std::vector<cv::String> &filenames);

/*
 * Create the output directory if it does not exist.
 * @param output_dir: the output directory
 */
void createFolder(cv::String output_dir);

/*
 * Check if the image is read successfully, if not, throw an exception.
 * @param data: the data pointer of the image
 * @param filename: the filename of the image
 */
void checkImg(uchar* data, cv::String filename);

/*
 * Get the minimum int value of four float numbers.
 * @param a, b, c, d: the four float numbers
 * @return: the minimum of the four float numbers
 */
int floorMin(float a, float b, float c, float d);

/*
 * Get the maximum int value of four float numbers.
 * @param a, b, c, d: the four float numbers
 * @return: the maximum of the four float numbers
 */
int ceilMax(float a, float b, float c, float d);

/*
 * Get the mask of the image, by thresholding the grayscale image.
 * Support dilate and erode the mask.
 * If dilate_size or erode_size is not -1, then dilate or erode the mask.
 * @param img: the image
 * @param mask: the mask of the image
 * @param dilate_size: the size of the dilate kernel, -1 means no dilate
 * @param erode_size: the size of the erode kernel, -1 means no erode
 */
void getMask(cv::Mat &img, cv::Mat &mask, int dilate_size = -1, int erode_size = -1);

/*
 * Get the minimum enclosing rectangle of the mask.
 * @param mask: the mask
 * @param rect: the minimum enclosing rectangle
 */
void getMinEnclosingRect(cv::Mat &mask, cv::Rect &rect);

/*
 * Get the maximum inner rectangle of the mask.
 * Refer to https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
 * @param mask: the mask
 * @param rect: the maximum inner rectangle
 */
void getMaxInnerRect(cv::Mat &mask, cv::Rect &rect);

#endif