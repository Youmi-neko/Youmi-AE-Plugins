# YO Tebure Shake - CMake build (Windows x64 .aex / macOS Universal .plugin)
#
#   macOS:   cmake -S . -B build -DAE_SDK=<...>/AfterEffectsSDK_26.5_mac/Examples -DCMAKE_BUILD_TYPE=Release
#            cmake --build build -j
#   Windows: (VS 2022 x64 Native Tools prompt)
#            cmake -S . -B build -G Ninja -DAE_SDK=<...>\AfterEffectsSDK_26.5_win\Examples -DCMAKE_BUILD_TYPE=Release
#            cmake --build build -j
#
# TB_VARIANT: default (SSE2 + AVX dispatch on x86_64, scalar on arm64) | noavx | scalar (no intrinsics anywhere).
cmake_minimum_required(VERSION 3.21)
if(APPLE OR CMAKE_HOST_APPLE)
  if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "universal: Apple Silicon + Intel")
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "minimum macOS")
  endif()
endif()
project(Tebure VERSION 0.3.1.1 LANGUAGES C CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
include(cmake/AEPlugin.cmake)

set(TB_VARIANT "default" CACHE STRING "default | noavx | scalar")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ENGINE_SRC ${SRC}/tb_model.cpp ${SRC}/tb_plan.cpp ${SRC}/tb_warp.cpp ${SRC}/tb_presets.cpp ${SRC}/tb_thumbs.cpp ${SRC}/tb_warp_avx.cpp)
if(WIN32)
  set(PLAT_SRC ${SRC}/platform/plat_win.cpp)
elseif(APPLE)
  set(PLAT_SRC ${SRC}/platform/plat_mac.cpp)
else()
  message(FATAL_ERROR "only Windows and macOS have a platform layer (src/platform)")
endif()
ae_avx_source(${SRC}/tb_warp_avx.cpp)

set(TB_DEFS NDEBUG TB_STR_ENC=3)
if(TB_VARIANT STREQUAL "noavx")
  list(APPEND TB_DEFS TB_NO_AVX)
elseif(TB_VARIANT STREQUAL "scalar")
  list(APPEND TB_DEFS TB_FORCE_SCALAR)
endif()

function(tb_common tgt)
  ae_strict_fp(${tgt})
  target_include_directories(${tgt} PRIVATE ${SRC})
  target_compile_definitions(${tgt} PRIVATE ${TB_DEFS})
  if(APPLE)
    target_link_libraries(${tgt} PRIVATE "-framework CoreFoundation" iconv)
  endif()
endfunction()

add_library(tb_engine STATIC ${ENGINE_SRC} ${SRC}/tb_strings.cpp ${PLAT_SRC})
tb_common(tb_engine)
set_target_properties(tb_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set_target_properties(tb_engine PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

ae_add_effect(Tebure
  OUTPUT_NAME YO_Tebure
  PIPL ${SRC}/TeburePiPL.r
  RC ${SRC}/Tebure.rc
  BUNDLE_ID com.youmi.tebure
  VERSION 0.3.1.1
  SOURCES ${SRC}/Tebure.cpp)
tb_common(Tebure)
target_link_libraries(Tebure PRIVATE tb_engine)
