# 1. Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.20)

# 2. Name your project (this can be anything you want)
project(NumericalMethodsCPP)

# 3. Specify the C++ standard you want to use (C++17 is a solid, modern choice)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Tell CMake to create an executable program from your source files
# The first word is the name of the executable it will build.
# Everything after that is the list of .cpp files it needs to compile.
add_executable(NumericalMethodsCPP
    main.cpp
)