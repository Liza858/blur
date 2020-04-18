#include <iostream>
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


#define BIG_INT 1000000
#define COUNT_OF_ITERATIONS_IN_EDGE_DIFFUSION 200
  using std::vector;
    using std::string;
    using std::ifstream;
    using std::ofstream;
    using std::cin;
    using std::cout;
    using std::endl;
    using std::map;
    using std::pair;

class image {

    private:
    
        string path;
        cv::Mat img;
        cv::Mat image_black;
        cv::Mat image_with_edges;
        cv::Mat image_test;
        int w;
        int h;
        pair<double, double>* gradient_matrix;
        int matrix1[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, -1, 0}};
        int matrix2[3][3] = {{0, 0, 0}, {1, 0, -1}, {0, 0, 0}};
        
    public:
    
        image(string image_file) {
            path = image_file;
            img = cv::imread(image_file, CV_LOAD_IMAGE_GRAYSCALE);
            w = img.cols;
            h = img.rows;  
        }

        void sobel_detect() {
              for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                   cout << static_cast<int>(img.at<uint8_t>(i, j)) << " " ;
                }
                cout << endl;
              }
             image_black = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
             for (int i = 1; i < h-1; i++) {
                for (int j = 1; j < w-1; j++) {
                    int Gx = 0;
                    int Gy = 0;
                    for (int p = -1; p <= 1; p++) {
                        for (int q = -1; q <= 1; q++) {
                            Gx += matrix2[p+1][q+1] * static_cast<int>(img.at<uint8_t>(i + p, j + q));
                            Gy += matrix1[p+1][q+1] * static_cast<int>(img.at<uint8_t>(i + p, j + q));
                        }
                    }
                    double gradient = sqrt(Gx*Gx + Gy*Gy);
                    if ((int)gradient > 255) {
                        gradient = 255;
                    }
                    image_black.at<uint8_t>(i, j) = (int)gradient;
                 }
             }
             cv::imwrite("vf.png", image_black);
        }
        
        
        vector<double> find_next_pixel(int gx, int gy, vector<double> pixel, int direction) {
            double step = 1.0;
            vector<double> next;
            if (gx == 0 && gy == 0) {
                return pixel;
            }
            if (abs(gx) > abs(gy)) {
                if (gx >= 0) {
                    step = step * direction;
                } else {
                    step = -step * direction;
                }
                double k = gy / gx;
                double b = pixel[0] - k * pixel[1]; // b = y - k*x
                next = vector<double>{k * (pixel[1] + step) + b, (pixel[1] + step)};
            } else {
                if (gy >= 0) {
                    step = step * direction;
                } else {
                    step = -step * direction;
                }
                double k = gx / gy;
                double b = pixel[1] - k * pixel[0];
                next = vector<double>{(pixel[0] + step), k * (pixel[0] + step) + b};
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
                            return BIG_INT;
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
                            return BIG_INT;
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
                            return BIG_INT;
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
                            return BIG_INT;
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

        void compute_vertical_or_horizontal_eages(bool isVertical) {
            vector<vector<double>> eages_img;
            vector<double> eages_width;
            vector<double> vec;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int width = 1;
                    if (isVertical) {
                        int grad =  static_cast<int>(img.at<uint8_t>(i+1, j)) - static_cast<int>(img.at<uint8_t>(i-1, j));
                        width += compute_vertical_eage_width(i, j, grad, true, img);
                        width += compute_vertical_eage_width(i, j, grad, false, img);
                    } else {
                        int grad =  static_cast<int>(img.at<uint8_t>(i, j+1)) - static_cast<int>(img.at<uint8_t>(i, j-1));
                        width += compute_horizontal_eage_width(i, j, grad, true, img);
                        width += compute_horizontal_eage_width(i, j, grad, false, img);
                    }
                    cout << "width " << width << endl;
                    double w = 1.0 / width;
                    eages_width.push_back(w);
                    vec.push_back(w);    
                }  
                eages_img.push_back(vec);
                vec.clear();
            }

            std::sort(eages_width.begin(), eages_width.end());
            int index = eages_width.size() / 100 * 90;
            double pivot = eages_width[index];
            image_with_edges = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                     double w = eages_img[i][j];
                     if (w > pivot) {
                          image_with_edges.at<uint8_t>(i, j) = (int)255*(1 - exp(-10*(w-pivot)));
                     }
                }
            }
            cv::imwrite("./step1.png", image_with_edges);
        }


        
        void compute_eages() {
            vector<vector<double>> eages_w;
            vector<double> eages_width;
            cv::Mat img_src;
            cv::copyMakeBorder(img, img_src, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
            for (int i = 1; i < h; i++) {
                vector<double> vec;
                for (int j = 1; j < w; j++) {
                     int gx = static_cast<int>(img_src.at<uint8_t>(i, j+1)) - static_cast<int>(img_src.at<uint8_t>(i, j-1));
                     int gy = static_cast<int>(img_src.at<uint8_t>(i+1, j)) - static_cast<int>(img_src.at<uint8_t>(i-1, j));
                     int counter = 1;
                     vector<double> pixel{i, j};
                     vector<double> next = pixel;
                     int color_pixel = static_cast<int>(img_src.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                     int color_next = color_pixel;
                     while(true) {
                         color_pixel = static_cast<int>(img_src.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                         next = find_next_pixel(gx, gy, pixel, 1);
                         if (next == pixel) {
                             counter = BIG_INT;
                             break;
                         }
                         color_next = static_cast<int>(img_src.at<uint8_t>((int)next[0], (int)next[1]));
                         //cout << "pixel " <<  pixel[0] << " " << pixel[1] << " " << color_pixel << endl;
                         //cout << "next " << next[0] << " " << next[1] << " " << color_next <<endl;
                         pixel = next;
                         if (color_next >= color_pixel) {
                             counter++;
                         } else {
                             break;
                         }
                     }
                     pixel = vector<double>{i, j};
                     next = pixel;
                     color_pixel = static_cast<int>(img_src.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                     color_next = color_pixel;
                     while(true) {
                         color_pixel = static_cast<int>(img_src.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                         next = find_next_pixel(gx, gy, pixel, -1);
                         if (next == pixel) {
                             counter = BIG_INT;
                             break;
                         }
                         pixel = next;
                         if (color_next <= color_pixel) {
                             counter++;
                         } else {
                             break;
                         }
                     }
                        
                     double w = 1.0 / counter;
                     //cout << w  << "  " << j << " "  << i << endl;
                     eages_width.push_back(w);
                     vec.push_back(w);              
                }
                eages_w.push_back(vec);
            }

            std::sort(eages_width.begin(), eages_width.end());
            
            size_t index = eages_width.size() / 100 * 95;
            double pivot = eages_width[index];
            image_with_edges = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
            for (int i = 0; i < h-1; i++) {
                for (int j = 0; j < w-1; j++) {
                     double w = eages_w[i][j];
                     if (w > pivot) {
                          cout << i << " " << j << endl;
                          image_with_edges.at<uint8_t>(i, j) = (int)255*(1 - exp(-10*(w-pivot)));
                     }
                }
            }
            cv::imwrite("./step1.png", image_with_edges);
        }

        void compute_edge_diffusion() {
            cv::Mat imm = image_with_edges;
            //cv::copyMakeBorder(imm, imm, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0);
            int w = imm.cols;
            int h = imm.rows;
            cv::Mat image_diffusion(imm.rows, imm.cols ,CV_32S, cv::Scalar(0,0,0));
            image_test = cv::Mat(imm.rows, imm.cols ,CV_32S, cv::Scalar(0,0,0));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    image_diffusion.at<int32_t>(i, j) = static_cast<int32_t>(imm.at<uint8_t>(i, j));
                    image_test.at<int32_t>(i, j) = static_cast<int32_t>(imm.at<uint8_t>(i, j));
                }
            }
            bool flag = true;
            double f;
            int min = 100000000;
            int max = -10000000;
            for (int k = 0; k < COUNT_OF_ITERATIONS_IN_EDGE_DIFFUSION; k++) {
                for (int i = 1; i < h - 3; i++) {
                    for (int j = 1; j < w - 3; j++) {
                        double dx = 1.0;
                        double dy = 1.0;
                        double u = image_test.at<int32_t>(i, j);
                        double u1 = image_test.at<int32_t>(i-1, j);
                        double u2 = image_test.at<int32_t>(i+1, j);
                        double u3 = image_test.at<int32_t>(i, j-1);
                        double u4 = image_test.at<int32_t>(i, j+1); 
                        f = image_diffusion.at<int32_t>(i, j);
                       // cout << u1 << " " << u2 << " " << u3 << " " << u4 <<  " " <<  f <<endl;
                        double result = ((u1 + u2) * dy*dy) / (2 * (dx*dx + dy*dy)) +
                                        ((u3 + u4) * dx*dx) / (2 * (dx*dx + dy*dy)) +
                                        (dx*dx * dy*dy * f) / (2 * (dx*dx + dy*dy));

                         if (i > 3 && j > 3 &&  i < h-7 && j < w-7) {
                         if (static_cast<int32_t>(result) > max) {
                            
                             max = static_cast<int32_t>(result);
                              //cout <<"max "<< i << " " << j << " " << max << endl;
                         }
                         if (static_cast<int32_t>(result) < min) {
                             min = static_cast<int32_t>(result);
                              //cout <<"min "<< i << " " << j << " " << min << endl;
                         }}
                       // cout << static_cast<int32_t>(result) << "*" << k << endl;
                        
                        image_test.at<int32_t>(i, j) = static_cast<int32_t>(result);
                    }  
                  //cout <<  "******" << endl;
                }
                //flag = false;
                //for (int i = 1; i < h - 3; i++) {
                    //for (int j = 1; j < w - 3; j++) {                  
                        //cout << (max - min) << "maxmin " << min << " " << max << endl;
                       // cout << image_diffusion.at<int32_t>(i, j) << endl;
                        //image_diffusion.at<int32_t>(i, j) = (image_diffusion.at<int32_t>(i, j)) * 255 / (max - min) - min;
                       // cout << image_diffusion.at<int32_t>(i, j) << endl;
                   // }
               // }
               // image_test = image_diffusion;
              }

         for (int i = 0; i < h - 2; i++) {
             for (int j = 0; j < w - 2; j++) {
                if (image_test.at<int32_t>(i, j) < 0) {
                  image_test.at<int32_t>(i, j) = 0;
                } else {
                 image_test.at<int32_t>(i, j) = (image_test.at<int32_t>(i, j)) * 255 / (max - min) - min;
                }       
             }
         }
         cv::imwrite("./step2.png", image_test);
        }

        void gu() {
           cv::Mat I = cv::imread(path, CV_LOAD_IMAGE_COLOR);
           cv::Mat II = cv::imread("./step2.png", CV_LOAD_IMAGE_GRAYSCALE);
           //cv::copyMakeBorder(II, II, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0 );
           
           int r = 10;
           double eps = 0.1;

           eps *= 255 * 255;   

           cv::Mat q = fastGuidedFilter(I, II,  r, eps);
           cv::imwrite("./step3.png", q);
        }

        
};



int main() {

    image im("./od.png");
    //im.sobel_detect();
    im.compute_eages();
    im.compute_edge_diffusion();
    im.gu();
    



}
