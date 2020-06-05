#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <map>
#include <utility>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
#include "fastguidedfilter.h"
#include <dirent.h>
#include <experimental/filesystem>

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::pair;
namespace fs =  std::experimental::filesystem;

class image {


    private:
    
        string directory;
        string file_name;
        string output_directory;
        fs::path computing_edges_directory;
        fs::path edges_diffusion_directory;
        fs::path guided_filter_directory;
        cv::Mat source_image;
        cv::Mat image_with_edges;
        cv::Mat image_with_edges_diffusion;
        cv::Mat image_with_guided_filter;
        int w;
        int h;

        
    public:
    
        image(string dir, string file, string output_dir = "./results") {
            output_directory = output_dir;
            directory = dir;
            file_name = file;
            create_directories();
            fs::path path = fs::path(directory) /= fs::path(file_name);
            source_image = cv::imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
            w = source_image.cols;
            h = source_image.rows;  
        }

        string get_guided_filter_directory() {
            return (fs::path(output_directory) /= fs::path("guided_filter_results")).string();
        }

        void create_directories() {
            computing_edges_directory = fs::path("computing_edges_results");
            edges_diffusion_directory = fs::path("edges_diffusion_results");
            guided_filter_directory = fs::path("guided_filter_results");
            fs::create_directory(output_directory);
            fs::create_directory(fs::path(output_directory) /= computing_edges_directory);
            fs::create_directory(fs::path(output_directory) /= edges_diffusion_directory);
            fs::create_directory(fs::path(output_directory) /= guided_filter_directory);
        }

        void gaus_filter(double sigma) {
            fs::path path = fs::path(output_directory) /= fs::path(file_name);
            cv::imwrite(path.string(), source_image);
            cv::GaussianBlur(source_image, source_image, cv::Size(9, 9), sigma);
            fs::path output_path = fs::path(output_directory) /= fs::path("gaus" + file_name);
            cv::imwrite(output_path.string(), source_image);
        }

        void compress_image() {
            h = h % 2 == 0 ? h / 2 : (h - 1) / 2;
            w = w % 2 == 0 ? w / 2 : (w - 1) / 2;
            cv::Mat result = cv::Mat(h, w, CV_8U, cv::Scalar(0,0,0));
            for (int i = 0; i < source_image.rows; i+=2) {
                for (int j = 0; j < source_image.cols; j+=2) {
                    int pix1 = static_cast<int>(source_image.at<uint8_t>(i, j));
                    int pix2 = static_cast<int>(source_image.at<uint8_t>(i, j+1));
                    int pix3 = static_cast<int>(source_image.at<uint8_t>(i+1, j));
                    int pix4 = static_cast<int>(source_image.at<uint8_t>(i+1, j+1)); 
                    result.at<uint8_t>(i / 2, j / 2) = (pix1 + pix2 + pix3 + pix4) / 4;
                }
            }
            source_image = result;
            fs::path path = fs::path(output_directory) /= fs::path(file_name);
            cv::imwrite(path.string(), result);
        }

       
        vector<int> find_next_pixel(int gx, int gy, vector<int> pixel, bool is_positive_direction) {
            int step = 1;
            if (!is_positive_direction) {
                step = -step;
            }
            vector<int> next;
            if (gx == 0 && gy == 0) {
                return pixel;
            }
            if (abs(gx) > abs(gy)) {
                if (gx >= 0) {
                    step = step;
                } else {
                    step = -step;
                }
                double k = gy / gx;
                double b = pixel[0] - k * pixel[1]; // b = y - k*x
                int i = floor(k * (pixel[1] + step) + b);
                int j = pixel[1] + step;
                next = vector<int>{i, j};
            } else {
                if (gy >= 0) {
                    step = step;
                } else {
                    step = -step;
                }
                double k = gx / gy;
                double b = pixel[1] - k * pixel[0];
                int i = pixel[0] + step;
                int j = floor(k * (pixel[0] + step) + b);
                next = vector<int>{i, j};
            }
            if (next[0] > h-1 || next[1] > w-1 || next[0] < 0  || next[1] < 0) {
                next = pixel;
            }
            return next;
        }

        int compute_vertical_eage_width(int i, int j, int grad, int isTop, cv::Mat& image) {
            int width = 0;
            if (grad > 0 && isTop || grad < 0 && !isTop) {
                int prev_pixel = static_cast<int>(image.at<uint8_t>(i, j));
                while(i - 1 >= 0) {
                    int next_pixel = static_cast<int>(image.at<uint8_t>(i-1, j));
                    if (next_pixel >= prev_pixel) {
                        if (i - 1 == 0) {
                            return std::numeric_limits<int>::max();
                        }
                        width++;
                    } else {
                        break;
                    }
                    prev_pixel = next_pixel; 
                    i--;
                }
            } else {
                int prev_pixel = static_cast<int>(image.at<uint8_t>(i, j));
                while(i + 1 <= h - 1) {
                    int next_pixel = static_cast<int>(image.at<uint8_t>(i+1, j));
                    if (next_pixel >= prev_pixel) {
                        if (i + 1 == h - 1) {
                            return std::numeric_limits<int>::max();
                        }
                        width++;
                    } else {
                        break;
                    }
                    prev_pixel = next_pixel; 
                    i++;
                }
             }
             return width;
        }

        int compute_horizontal_eage_width(int i, int j, int grad, int isLeft, cv::Mat& image) {
            int width = 0;
            if (grad > 0 && isLeft || grad < 0 && !isLeft) {
                int prev_pixel = static_cast<int>(image.at<uint8_t>(i, j));
                while(j - 1 >= 0) {
                    int next_pixel = static_cast<int>(image.at<uint8_t>(i, j-1));
                    if (next_pixel >= prev_pixel) {
                        if (j - 1 == 0) {
                            return std::numeric_limits<int>::max();
                        }
                        width++;
                    } else {
                        break;
                    }
                    prev_pixel = next_pixel; 
                    j--;
                }
            } else {
                int prev_pixel = static_cast<int>(image.at<uint8_t>(i, j));
                while(j + 1 <= w - 1) {
                    int next_pixel = static_cast<int>(image.at<uint8_t>(i, j+1));
                    if (next_pixel >= prev_pixel) {
                        if (j + 1 == w - 1) {
                            return std::numeric_limits<int>::max();
                        }
                        width++;
                    } else {
                        break;
                    }
                    prev_pixel = next_pixel; 
                    j++;
                }
             }
             return width;
        }

        void compute_vertical_or_horizontal_edges(bool isHorizontal) {
            vector<vector<double>> edges_img;
            vector<double> edges_width;
            vector<double> vec;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int width = 1;
                    if (isHorizontal) {
                        int grad =  static_cast<int>(source_image.at<uint8_t>(i+1, j)) - static_cast<int>(source_image.at<uint8_t>(i-1, j));
                        width += compute_vertical_eage_width(i, j, grad, true, source_image);
                        width += compute_vertical_eage_width(i, j, grad, false, source_image);
                    } else {
                        int grad =  static_cast<int>(source_image.at<uint8_t>(i, j+1)) - static_cast<int>(source_image.at<uint8_t>(i, j-1));
                        width += compute_horizontal_eage_width(i, j, grad, true, source_image);
                        width += compute_horizontal_eage_width(i, j, grad, false, source_image);
                    }
                    double w = 1.0 / width;
                    edges_width.push_back(w);
                    vec.push_back(w);    
                }  
                edges_img.push_back(vec);
                vec.clear();
            }

            std::sort(edges_width.begin(), edges_width.end());
            int index = edges_width.size() / 100 * 95;
            double pivot = edges_width[index];
            image_with_edges = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                     double w = edges_img[i][j];
                     if (w > pivot) {
                          image_with_edges.at<uint8_t>(i, j) = (int)255*(1 - exp(-10*(w-pivot)));
                     }
                }
            }
            fs::path path = fs::path(output_directory) /= edges_diffusion_directory /= fs::path(file_name);
            cv::imwrite(path.string(), image_with_edges_diffusion);
        }

       
        int compute_edge_width(const cv::Mat& border_image, int grad_x, int grad_y, 
                                   vector<int> current_pixel, bool is_positive_direction,
                                   int steps_limit) {
            int edge_width = 0;
            vector<int> next_pixel = current_pixel;
            int counter = 0;
            while(true) {
                counter += 1;
                if (counter > steps_limit) { 
                    edge_width = std::numeric_limits<int>::max();
                    break;
                }
                int color_current_pixel = static_cast<int>(border_image.at<uint8_t>(current_pixel[0], current_pixel[1]));
                next_pixel = find_next_pixel(grad_x, grad_y, current_pixel, is_positive_direction);
                if (next_pixel == current_pixel) {
                    edge_width = std::numeric_limits<int>::max();
                    break;
                }
                int color_next_pixel = static_cast<int>(border_image.at<uint8_t>(next_pixel[0], next_pixel[1]));
                current_pixel = next_pixel;
                if (is_positive_direction) {
                    if (color_next_pixel >= color_current_pixel) {
                        edge_width++;
                    } else {
                        break;
                    }
                } else {
                    if (color_next_pixel <= color_current_pixel) {
                        edge_width++;
                    } else {
                        break;
                    }
                }
            }  
            return edge_width;
        }

        vector<int> compute_edges(int limit_edge_width = 3, int steps_limit = std::numeric_limits<int>::max()) {
            vector<vector<int>> edges;
            vector<int> edges_width;
            cv::Mat border_image;
            cv::copyMakeBorder(source_image, border_image, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
            for (int i = 1; i < border_image.rows - 1; i++) {
                vector<int> vec;
                for (int j = 1; j < border_image.cols - 1; j++) {

                     int grad_x = static_cast<int>(border_image.at<uint8_t>(i, j+1)) - static_cast<int>(border_image.at<uint8_t>(i, j-1));
                     int grad_y = static_cast<int>(border_image.at<uint8_t>(i+1, j)) - static_cast<int>(border_image.at<uint8_t>(i-1, j));

                     vector<int> pixel{i, j};
                     int edge_width = 1;
                     int width1 = compute_edge_width(border_image, grad_x, grad_y, pixel, true, steps_limit);
                     int width2 = compute_edge_width(border_image, grad_x, grad_y, pixel, false, steps_limit);
                     if (width1 == std::numeric_limits<int>::max() || width2 == std::numeric_limits<int>::max()) {
                         edge_width = std::numeric_limits<int>::max();
                     } else {
                         edge_width += width1;
                         edge_width += width2;
                     }
                     
                     if (abs(grad_x) < 5 && abs(grad_y) < 5) {
                        edge_width = std::numeric_limits<int>::max();
                     }

                     edges_width.push_back(edge_width);
                     vec.push_back(edge_width);              
                }
                edges.push_back(vec);
            }

           /* std::sort(edges_width.begin(), edges_width.end());
            int index = edges_width.size() / 100 * 3;
            int pivot = edges_width[index]; */

            image_with_edges = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int w = edges[i][j];
                    if (w <= limit_edge_width) {
                        double reverse_w = 1.0 / w;
                        image_with_edges.at<uint8_t>(i, j) = floor(255 * (1 - exp(-10 * (reverse_w - 0))));
                    }
                }
            }
            fs::path path = fs::path(output_directory) /= computing_edges_directory /= fs::path(file_name);
            cv::imwrite(path.string(), image_with_edges);
            return edges_width;
        }

        void compute_edge_diffusion(int count_of_iterations=200) {
            cv::Mat imm = image_with_edges;
            int w = imm.cols;
            int h = imm.rows;
            cv::Mat image_diffusion(imm.rows, imm.cols ,CV_32S, cv::Scalar(0,0,0));
            image_with_edges_diffusion = cv::Mat(imm.rows, imm.cols ,CV_32S, cv::Scalar(0,0,0));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    image_diffusion.at<int32_t>(i, j) = static_cast<int32_t>(imm.at<uint8_t>(i, j));
                    image_with_edges_diffusion.at<int32_t>(i, j) = static_cast<int32_t>(imm.at<uint8_t>(i, j));
                }
            }
            int min = std::numeric_limits<int>::max();
            int max = std::numeric_limits<int>::min();
            for (int k = 0; k < count_of_iterations; k++) {
                for (int i = 1; i < h - 1; i++) {
                    for (int j = 1; j < w - 1; j++) {
                        double dx = 1.0;
                        double dy = 1.0;
                        double u = image_with_edges_diffusion.at<int32_t>(i, j);
                        double u1 = image_with_edges_diffusion.at<int32_t>(i-1, j);
                        double u2 = image_with_edges_diffusion.at<int32_t>(i+1, j);
                        double u3 = image_with_edges_diffusion.at<int32_t>(i, j-1);
                        double u4 = image_with_edges_diffusion.at<int32_t>(i, j+1); 
                        double f = image_diffusion.at<int32_t>(i, j);
                        double result = ((u1 + u2) * dy*dy) / (2 * (dx*dx + dy*dy)) +
                                        ((u3 + u4) * dx*dx) / (2 * (dx*dx + dy*dy)) +
                                        (dx*dx * dy*dy * f) / (2 * (dx*dx + dy*dy));
                        int int_result = static_cast<int>(round(result));
                        if (i > 3 && j > 3 &&  i < h - 7 && j < w - 7) {
                            if (int_result > max) {
                                max = int_result;
                            }
                            if (int_result < min) {
                                min = int_result;
                            }
                        }
                        image_with_edges_diffusion.at<int32_t>(i, j) = static_cast<int32_t>(int_result);
                    }  
                }
            }
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if (image_with_edges_diffusion.at<int32_t>(i, j) < 0) {
                        image_with_edges_diffusion.at<int32_t>(i, j) = 0;
                    } else {
                        int32_t pixel = image_with_edges_diffusion.at<int32_t>(i, j);
                        image_with_edges_diffusion.at<int32_t>(i, j) = pixel * 255 / (max - min) - min;
                    }       
                }
            }
            fs::path path = fs::path(output_directory) /= edges_diffusion_directory /= fs::path(file_name);
            cv::imwrite(path.string(), image_with_edges_diffusion);
        }

        void guided_filter(double r = 0, double eps = 0.01) {   
           if (r == 0) {
              r = 30.0 / ( sqrt(15980544) / sqrt(h*w) );
           }  
           cout << r << endl;    
           eps *= 255 * 255;   
           image_with_guided_filter = fastGuidedFilter(source_image, image_with_edges_diffusion,  r, eps);
           fs::path path = fs::path(output_directory) /= guided_filter_directory /= fs::path(file_name);
           cv::imwrite(path.string(), image_with_guided_filter);
        }
       
};
