# SPDX-License-Identifier: GPL-2.0-only
# nvimx296camerasrc - GStreamer source element: IMX296 raw V4L2 capture +
# fused CUDA ISP (black level -> debayer -> WB*CCM -> tone map -> NV12)
# outputting memory:NVMM buffers. Build natively on the Jetson (JetPack 6).
cmake_minimum_required(VERSION 3.18)
project(nvimx296camerasrc LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)   # Orin (Ampere iGPU)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
find_package(CUDAToolkit REQUIRED)

set(JETSON_MM_API /usr/src/jetson_multimedia_api)
set(TEGRA_LIB_DIRS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)

add_library(gstnvimx296camerasrc SHARED
  src/gstnvimx296camerasrc.cpp
  src/imx296_isp.cu
  src/tuning.cpp
)
target_include_directories(gstnvimx296camerasrc PRIVATE
  ${GST_INCLUDE_DIRS} ${JETSON_MM_API}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(gstnvimx296camerasrc PRIVATE ${TEGRA_LIB_DIRS})
target_link_libraries(gstnvimx296camerasrc
  ${GST_LIBRARIES} nvbufsurface EGL CUDA::cudart CUDA::cuda_driver
)
target_compile_options(gstnvimx296camerasrc PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)
set_target_properties(gstnvimx296camerasrc PROPERTIES
  PREFIX "lib" CUDA_SEPARABLE_COMPILATION OFF)

# Golden-test tool: raw frame file -> kernel -> RGB8 dump (no GStreamer, no NVMM)
add_executable(imx296_kernel_test tools/imx296_kernel_test.cpp src/imx296_isp.cu src/tuning.cpp)
target_include_directories(imx296_kernel_test PRIVATE
  ${JETSON_MM_API}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(imx296_kernel_test CUDA::cudart CUDA::cuda_driver)

install(TARGETS gstnvimx296camerasrc LIBRARY DESTINATION lib/gstreamer-1.0)
