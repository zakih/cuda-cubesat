cmake_minimum_required(VERSION 3.18)
project(cubesat_thermal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---- OpenMP (required for the OpenMP backend) ------------------------------
find_package(OpenMP REQUIRED)

# ---- CUDA (optional) -------------------------------------------------------
# If a CUDA compiler is found we build the GPU backends and define HAVE_CUDA.
# Otherwise the program builds CPU-only (serial + OpenMP) and the CUDA columns
# report as unavailable.
include(CheckLanguage)
check_language(CUDA)

set(CPU_SOURCES
    src/main.cpp
    src/solver_serial.cpp
    src/solver_omp.cpp
)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 61 = Pascal (GTX 1050). Override with -DCMAKE_CUDA_ARCHITECTURES=75 etc.
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif()
    add_executable(cubesat_thermal ${CPU_SOURCES} src/solver_cuda.cu)
    target_compile_definitions(cubesat_thermal PRIVATE HAVE_CUDA)
    set_target_properties(cubesat_thermal PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "CUDA found: building GPU backends (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
    add_executable(cubesat_thermal ${CPU_SOURCES})
    message(STATUS "CUDA NOT found: building CPU-only (serial + OpenMP)")
endif()

target_include_directories(cubesat_thermal PRIVATE src)
target_link_libraries(cubesat_thermal PRIVATE OpenMP::OpenMP_CXX)
