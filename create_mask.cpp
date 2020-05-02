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


int find_limit(const std::string& dir_path, double coefficient) {

    using namespace std;

    struct dirent *entry;
    DIR *dir = opendir(dir_path.c_str());
    int global_sum = 0;
    int global_counter = 0;
    int global_maxx = 0;
    while ((entry = readdir(dir)) != nullptr) {
        string file = entry->d_name;
    	if (file != "." && file != "..") {
      	    int sum = 0;
      	    int counter = 1;
      	    cv::Mat img = cv::imread(dir_path + "/" + file, CV_LOAD_IMAGE_GRAYSCALE);
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

void create_binary_masks(const std::string& src_dir, const std::string& dst_dir, int limit) {

    using namespace std;

    struct dirent *entry;
    DIR *dir = opendir(src_dir.c_str());
    while ((entry = readdir(dir)) != nullptr) {
        string file = entry->d_name;
    	if (file != "." && file != "..") {
      	    cv::Mat img = cv::imread(src_dir + "/" + file, CV_LOAD_IMAGE_GRAYSCALE);
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
            cv::imwrite(dst_dir + "/" + file, result);
	    }
	}
    closedir(dir);
}


int main() {

    int limit = find_limit("./step3", 0);
    std::cout << limit << std::endl;
    create_binary_masks("/step3", "./mask", limit);
    return 0;
}
