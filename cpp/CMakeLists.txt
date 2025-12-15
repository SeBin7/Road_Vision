cmake_minimum_required(VERSION 3.18)
project(road_vision_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Allow user override: -DRV_VENV_ROOT=/path/to/.xpu
set(RV_VENV_ROOT "" CACHE PATH "Venv root to use for RPATH (e.g., /home/.../Road_Vision/.xpu)")
if(RV_VENV_ROOT STREQUAL "")
  if(DEFINED ENV{VIRTUAL_ENV})
    set(RV_VENV_ROOT $ENV{VIRTUAL_ENV})
  else()
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pathlib; import sys; print(pathlib.Path(sys.executable).resolve().parents[1])"
      OUTPUT_VARIABLE RV_VENV_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
endif()

message(STATUS "RV_VENV_ROOT = ${RV_VENV_ROOT}")
set(VENV_LIB_DIR "${RV_VENV_ROOT}/lib")

# Torch / OpenCV
# IMPORTANT: pass Torch_DIR explicitly:
#   -DTorch_DIR=$VENV/lib/python3.12/site-packages/torch/share/cmake/Torch
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

# Torch prefix -> torch/lib
get_filename_component(_TORCH_SHARE_CMAKE "${Torch_DIR}" DIRECTORY)          # .../torch/share/cmake
get_filename_component(_TORCH_SHARE      "${_TORCH_SHARE_CMAKE}" DIRECTORY)  # .../torch/share
get_filename_component(TORCH_PREFIX      "${_TORCH_SHARE}" DIRECTORY)        # .../torch
set(TORCH_LIB_DIR "${TORCH_PREFIX}/lib")

message(STATUS "TORCH_PREFIX  = ${TORCH_PREFIX}")
message(STATUS "TORCH_LIB_DIR = ${TORCH_LIB_DIR}")

# Runtime libs (SYCL + UR loader)
set(SYCL_RUNTIME_DIR "" CACHE PATH "Directory containing libsycl.so.8 (default: <venv>/lib)")
if(SYCL_RUNTIME_DIR STREQUAL "")
  set(SYCL_RUNTIME_DIR "${VENV_LIB_DIR}")
endif()

find_library(SYCL_RUNTIME_LIB NAMES sycl libsycl.so.8
  PATHS "${SYCL_RUNTIME_DIR}" "${SYCL_RUNTIME_DIR}/intel64"
        "/opt/intel/oneapi/compiler/latest/lib" "/opt/intel/oneapi/latest/lib"
  REQUIRED
)
message(STATUS "SYCL_RUNTIME_LIB = ${SYCL_RUNTIME_LIB}")

set(UR_LOADER_DIR "" CACHE PATH "Directory containing libur_loader.so (default: <venv>/lib)")
if(UR_LOADER_DIR STREQUAL "")
  set(UR_LOADER_DIR "${VENV_LIB_DIR}")
endif()

find_library(UR_LOADER_LIB NAMES ur_loader libur_loader.so
  PATHS "${UR_LOADER_DIR}" "${UR_LOADER_DIR}/intel64"
        "/opt/intel/oneapi/compiler/latest/lib" "/opt/intel/oneapi/latest/lib"
  REQUIRED
)
message(STATUS "UR_LOADER_LIB = ${UR_LOADER_LIB}")

# Targets
add_executable(road_vision_cpp src/main.cpp)
add_executable(road_vision_xpu
  src/road_vision_xpu.cpp
  src/rv/config.cpp
  src/rv/preprocess.cpp
  src/rv/overlay.cpp
  src/rv/pipeline.cpp
  src/rv/engine_factory.cpp
  src/rv/cpu_engine.cpp
  src/rv/xpu_engine.cpp
  src/rv/hybrid_engine.cpp
)

target_include_directories(road_vision_xpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

foreach(tgt IN ITEMS road_vision_cpp road_vision_xpu)
  target_compile_definitions(${tgt} PRIVATE TORCH_USE_XPU)
  target_compile_definitions(${tgt} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  target_compile_options(${tgt} PRIVATE -O3 -DNDEBUG)

  # Make link order robust and keep indirect deps on the command line.
  target_link_options(${tgt} PRIVATE "-Wl,--no-as-needed" "-Wl,--copy-dt-needed-entries")

  # IMPORTANT: put SYCL first, UR loader after it (GNU ld is order-sensitive).
  target_link_libraries(${tgt} PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${SYCL_RUNTIME_LIB}
    ${UR_LOADER_LIB}
  )

  # RPATH: prefer venv + torch libs first.
  set_target_properties(${tgt} PROPERTIES
    BUILD_RPATH "${RV_VENV_ROOT}/lib;${RV_VENV_ROOT};${TORCH_LIB_DIR};${SYCL_RUNTIME_DIR};${UR_LOADER_DIR}"
    INSTALL_RPATH "${RV_VENV_ROOT}/lib;${RV_VENV_ROOT};${TORCH_LIB_DIR};${SYCL_RUNTIME_DIR};${UR_LOADER_DIR}"
  )
endforeach()
