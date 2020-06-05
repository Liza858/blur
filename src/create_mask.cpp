#include "prog.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <dirent.h>
#include <sys/types.h>
#include <experimental/filesystem>
#include <stdexcept>
	


int find_limit(const std::string& dir_path, double coefficient) {

    using namespace std;

    struct dirent *entry;
    DIR *dir = opendir(dir_path.c_str());
    int global_sum = 0;
    int global_counter = 0;
    int global_maxx = 0;
    while ((entry = readdir(dir)) != nullptr) {
        string file_name = entry->d_name;
        fs::path file = fs::path(file_name);
    	if (!is_directory(file)) {
      	    int sum = 0;
      	    int counter = 1;
      	    cv::Mat img = cv::imread((fs::path(dir_path) /= file).string(), CV_LOAD_IMAGE_GRAYSCALE);
      	    int w = img.cols;
      	    int h = img.rows;  
            int maxx = 0; 
       	    for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
           	    int color = static_cast<int>(img.at<uint8_t>(i, j));
                    if (color > maxx) {
                        maxx = color;
                    }
                    if (color != 0) {
                        sum += color;
                        counter += 1;
             	    }
                }
            }
    	    int average = sum / counter;
            global_maxx += maxx;
            global_sum += average;
            global_counter += 1;
	}
    }
    closedir(dir);
    global_maxx = global_maxx / global_counter;
    int mean = global_sum / global_counter;
    int result = mean + (global_maxx - mean) * coefficient;
    return result;
}

void create_binary_mask(const std::string& input_dir, const std::string& file_name, const std::string& output_dir, int limit) {
    fs::path src_dir = fs::path(input_dir); 
    fs::path dst_dir = fs::path(output_dir);
    fs::path file = fs::path(file_name);
    fs::path input = src_dir /= file;
    if (!is_directory(input)) {
        fs::path masks_dir = fs::path(dst_dir) /= fs::path("masks");
        fs::create_directory(masks_dir);
        cv::Mat img = cv::imread(input.string(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat result = img;
      	int w = img.cols;
      	int h = img.rows;  
       	for (int i = 0; i < h; i++) {
	    for (int j = 0; j < w; j++) {
                int color = static_cast<int>(img.at<uint8_t>(i, j));
                if (color > limit) {
                    result.at<uint8_t>(i, j) = 255;
                } else {
                    result.at<uint8_t>(i, j) = 0;
                }
            }
    	}
        cv::imwrite((masks_dir /= file).string(), result);
    }
}


string detect_blur(const std::string& input_dir, const std::string& file_name, const std::string& output_dir) {
    image img = image(input_dir, file_name, output_dir);
    img.compute_edges(3, 20);
    img.compute_edge_diffusion();
    img.guided_filter();
    return img.get_guided_filter_directory();
}


int main(int argc, char *argv[]) {

    if (argc < 4) {
        throw std::runtime_error("wrong number of command line arguments");
    }

    int mod = 0;

    enum {DIRECTORY, FILE};


    if (string(argv[1]) != "-d" && string(argv[1]) != "-f") {
         throw std::runtime_error("wrong command line key");
    }
   
    if (string(argv[1]) == "-d") {
       mod = DIRECTORY;
    }

    if (string(argv[1]) == "-f") {
        mod = FILE;
    }

    switch(mod) {
        case DIRECTORY: {
             struct dirent *entry;
             DIR *dir = opendir(argv[2]);
             while ((entry = readdir(dir)) != nullptr) {
                 string file = entry->d_name;
                 fs::path input = fs::path(argv[2]) /= fs::path(file);
                 if (!is_directory(input)) {
                     string result_dir = detect_blur(argv[2], file, argv[3]);
                     cout << result_dir << endl;
                     create_binary_mask(result_dir, file, argv[3], 10);
                 }
             }
             closedir(dir);
             break;
        }
        case FILE: {
             string result_dir = detect_blur(argv[2], argv[3], argv[4]);
             create_binary_mask(result_dir, argv[3], argv[4], 10);
             break;
        }
    }
    return 0;
}
