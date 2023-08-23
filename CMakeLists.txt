# CMake要求的最小版本号
cmake_minimum_required(VERSION 3.17)
# 项目名称
project(MNIST_Libtorch_cpp)

# c++的版本，目前选择的是14
set(CMAKE_CXX_STANDARD 14)

# 切换cuda把cpu->cu117即可
# 设置CMake解释器所在目录，自己的libtorch所在位置
set(Torch_DIR D:/environment/PyTorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch/share/cmake/Torch)
set(OpenCV_DIR D:/environment/PyTorch/opencv-4.8.0/opencv/build)
# 配置头文件所要解析的具体路径，根据自己的libtorch所在文件夹选择
include_directories(D:/environment/PyTorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch/include)
include_directories(D:/environment/PyTorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch/include/torch/csrc/api/include)
find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

# 默认即可
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 配置项目下的C++文件，每新建一个C++文件要重新添加
add_executable(MNIST_Libtorch_cpp main.cpp src/cpp/model.cpp src/include/model.h)

# 添加外部库
target_link_libraries(MNIST_Libtorch_cpp ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET MNIST_Libtorch_cpp PROPERTY CXX_STANDARD 14)

# 将动态链接库拷贝到当前程序所在文件夹
if (MSVC)
    file(GLOB TORCH_DLLS D:/environment/PyTorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch/lib/*.dll)
    add_custom_command(TARGET MNIST_Libtorch_cpp  #目标项目
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:MNIST_Libtorch_cpp>) #项目路径

    file(GLOB OPENCV_DLLS D:/environment/PyTorch/opencv-4.8.0/opencv/build/x64/vc16/bin/*.dll)
    add_custom_command(TARGET MNIST_Libtorch_cpp  #目标项目
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:MNIST_Libtorch_cpp>) #项目路径
endif (MSVC)