all: main

opencv_include_path = ...
opencv_lib_path = ...

main: create_mask.o guidedfilter.o
	g++  -std=c++17 -O -I$(opencv_include_path) -I./libs/guided-filter -L$(opencv_lib_path)  create_mask.o guidedfilter.o -lopencv_highgui  -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lstdc++fs

create_mask.o: ./src/create_mask.cpp
	g++ -c -std=c++17 -O -I$(opencv_include_path) -I./libs/guided-filter -L$(opencv_lib_path)  ./src/create_mask.cpp -lopencv_highgui  -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lstdc++fs


guidedfilter.o: ./libs/guided-filter/guidedfilter.cpp
	g++ -c -std=c++17 -O -I$(opencv_include_path) -I./libs/guided-filter -L$(opencv_lib_path)  ./libs/guided-filter/guidedfilter.cpp -lopencv_highgui  -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lstdc++fs


clean:
	rm -rf *.o main
