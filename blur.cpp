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
    
        cv::Mat img;
        cv::Mat image_black;
        int w;
        int h;
        pair<double, double>* gradient_matrix;
        int matrix1[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, -1, 0}};
        int matrix2[3][3] = {{0, 0, 0}, {1, 0, -1}, {0, 0, 0}};
        
    public:
    
        image(string image_file) {
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
            double step = 0;
            vector<double> next;
            if (gx == 0 && gy == 0) {
                return pixel;
            }
            
            if (abs(gx) > abs(gy)) {
                if (gx >= 0) {
                    step = 0.2 * direction;
                } else {
                    step = -0.2 * direction;
                }
                double k = gy / gx;
                double b = pixel[0] - k * pixel[1]; // b = y - k*x
                next = pixel;
                double pix = pixel[1];
                bool predicate1 = (int)next[0] == (int)pixel[0] && (int)next[1] == (int)pixel[1];
                bool predicate2 = next[0] < h - 1 && next[1] < w - 1 && next[0] > 0 && next[1] > 0;
                while (predicate1 && predicate2) {
                    pix += step; 
                    next = vector<double>{k * pix + b, pix};
                    predicate1 = (int)next[0] == (int)pixel[0] && (int)next[1] == (int)pixel[1];
                    predicate2 = next[0] < h - 1 && next[1] < w - 1 && next[0] > 0 && next[1] > 0;
                    // cout << pixel[0] << " " << pixel[1] << " " << next[0] << " " << next[1] << " " << pix <<  endl;  
                }
            } else {
                if (gy >= 0) {
                    step = 0.2 * direction;
                } else {
                    step = -0.2 * direction;
                }
                double k = gx / gy;
                double b = pixel[1] - k * pixel[0]; // b = y - k*x
                next = pixel;
                double pix = pixel[0];
                bool predicate1 = (int)next[0] == (int)pixel[0] && (int)next[1] == (int)pixel[1];
                bool predicate2 = next[0] < h - 1 && next[1] < w - 1 && next[0] > 0 && next[1] > 0;
                while (predicate1 && predicate2) {
                    pix += step; 
                    next = vector<double>{pix, k * pix + b};
                    predicate1 = (int)next[0] == (int)pixel[0] && (int)next[1] == (int)pixel[1];
                    predicate2 = next[0] < h - 1 && next[1] < w - 1 && next[0] > 0 && next[1] > 0;
                   // cout << pixel[0] << " " << pixel[1] << " " << next[0] << " " << next[1] << " " << pix <<  endl;  
                }

            }
            return next;
        }
        
        void compute_eages() {
            cv::Mat im = cv::Mat(h, w ,CV_8U, cv::Scalar(0,0,0));
            for (int i = 1; i < h-1; i++) {
                for (int j = 1; j < w-1; j++) {
                     int gx = -static_cast<int>(img.at<uint8_t>(i, j-1)) + static_cast<int>(img.at<uint8_t>(i, j+1));
                     int gy = -static_cast<int>(img.at<uint8_t>(i-1, j)) + static_cast<int>(img.at<uint8_t>(i+1, j));
                     int counter = 1;
                     vector<double> pixel{i, j};
                     vector<double> next = pixel;
                     int color_pixel = static_cast<int>(img.at<uint8_t>(pixel[0], pixel[1]));
                     int color_next = color_pixel;
                     do {
                         color_pixel = static_cast<int>(img.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                         next = find_next_pixel(gx, gy, pixel, 1);
                         if (next == pixel) {
                              counter = 100000000;
                             break;
                         }
                         if (next[0] > h - 1 || next[1] > w - 1 || next[0] < 1  || next[1] < 1) {
                            counter = 100000000;
                            break;
                         }
                         color_next = static_cast<int>(img.at<uint8_t>(next[0], next[1]));
                         //cout << "pixel " <<  pixel[0] << " " << pixel[1] << " " << color_pixel << endl;
                         //cout << "next " << next[0] << " " << next[1] << " " << color_next <<endl;
                         pixel = next;
                         counter += 1;
                     } while (color_next >= color_pixel);
                     pixel = vector<double>{i, j};
                     next = pixel;
                     color_pixel = static_cast<int>(img.at<uint8_t>(pixel[0], pixel[1]));
                      color_next = color_pixel;
                    do {
                        
                         color_pixel = static_cast<int>(img.at<uint8_t>((int)pixel[0], (int)pixel[1]));
                         next = find_next_pixel(gx, gy, pixel, -1);
                         if (next == pixel) {
                              counter = 100000000;
                             break;
                         }
                         if (next[0] > h - 1 || next[1] > w - 1 || next[0] < 1  || next[1] < 1) {
                            counter = 100000000;
                            break;
                         }
                         color_next = static_cast<int>(img.at<uint8_t>(next[0], next[1]));
                         pixel = next;
                         counter += 1;
                     } while (color_next <= color_pixel);
                         
                     
                     double w = 1.0 / counter;
                     //cout << w  << "  " << j << " "  << i << endl;
                     
                     if (w > 0.155) {
                          
                          im.at<uint8_t>(i, j) = (int)255*(1 - exp(-10*(w-0.155)));
                     }
                     
                }
                
            }
        
            cv::imwrite("vgfbf.png", im);
        }
};













int main() {

    image im("od.png");
    im.sobel_detect();
    im.compute_eages();



}
