cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(MyLibTorchProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the conda environment's path to CMake's search list
# This is the key line that tells CMake where to look for Eigen
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

# Find the required packages
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)

# Create the executable
add_executable(my_app main.cpp utils.cpp)

# Link the libraries
target_link_libraries(my_app ${TORCH_LIBRARIES} Eigen3::Eigen)

set_property(TARGET my_app PROPERTY CXX_STANDARD 17)