# YO Map Bilinear Blur - CMake build (Windows x64 .aex / macOS Universal .plugin)
#
# macOS (CPU-only by default; Metal does not support 64-bit float compute):
#   cmake -S . -B build -DAE_SDK=/path/to/AfterEffectsSDK_26.5_mac/Examples -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Windows (with GPU stage: first run `cargo rustc --release --locked --lib --crate-type staticlib` in gpu/ with RUSTFLAGS="-C target-feature=+crt-static"):
#   cmake -S . -B build -G Ninja -DAE_SDK=...\AfterEffectsSDK_26.5_win\Examples -DMBB_GPU=ON -DMBB_GPULIB=...\mbbgpu.lib -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
cmake_minimum_required(VERSION 3.20)
if(APPLE OR CMAKE_HOST_APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "universal plug-in")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "oldest macOS")
endif()
project(MapBilinearBlur VERSION 0.3.1.1 LANGUAGES C CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
include(cmake/AEPlugin.cmake)
ae_require_sdk()

if(WIN32)
  set(_gpu_default ON)
else()
  set(_gpu_default OFF)
endif()
option(MBB_GPU "link the Rust + wgpu GPU engine (gpu/) instead of the CPU-only stub" ${_gpu_default})
set(MBB_GPULIB "" CACHE FILEPATH "GPU engine static library (mbbgpu.lib / universal libmbbgpu.a)")
if(MBB_GPU AND NOT EXISTS "${MBB_GPULIB}")
  message(FATAL_ERROR "MBB_GPU=ON needs -DMBB_GPULIB=<path to mbbgpu.lib / libmbbgpu.a>")
endif()

set(S ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MBB_ENGINE_SOURCES ${S}/mbb_core.cpp ${S}/mbb_strings.cpp ${S}/platform/plat_win.cpp ${S}/platform/plat_mac.cpp)

function(mbb_link_platform tgt with_gpu)
  target_include_directories(${tgt} PRIVATE ${S} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include)
  if(with_gpu AND MBB_GPU)
    target_link_libraries(${tgt} PRIVATE "${MBB_GPULIB}")
    if(WIN32)
      target_link_libraries(${tgt} PRIVATE kernel32 ntdll userenv ws2_32 dbghelp advapi32 user32 ole32 oleaut32 bcrypt shell32
                                           propsys runtimeobject synchronization dxgi d3d12)
    elseif(APPLE)
      target_link_libraries(${tgt} PRIVATE "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreGraphics" objc)
    endif()
  elseif(with_gpu)
    target_sources(${tgt} PRIVATE ${S}/mbbgpu_stub.cpp)
  endif()
  if(APPLE)
    target_link_libraries(${tgt} PRIVATE iconv "-framework CoreFoundation")
  elseif(WIN32)
    target_link_libraries(${tgt} PRIVATE version)
  endif()
endfunction()

ae_add_effect(mbb_ja
  OUTPUT_NAME YO_MapBilinearBlur_ja
  PIPL ${S}/MapBilinearBlurPiPL.r
  RC ${S}/MapBilinearBlur.rc
  BUNDLE_ID com.youmi.mapbilinearblur.ja
  VERSION 0.3.1.1
  SOURCES ${S}/MapBilinearBlur.cpp ${MBB_ENGINE_SOURCES})
target_compile_definitions(mbb_ja PRIVATE MBB_STR_ENC=3 NDEBUG)
mbb_link_platform(mbb_ja TRUE)
if(APPLE AND MBB_GPU AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md)
  add_custom_command(TARGET mbb_ja POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:mbb_ja>/Resources
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md $<TARGET_BUNDLE_CONTENT_DIR:mbb_ja>/Resources/THIRD_PARTY_NOTICES.md
    COMMENT "bundle: third-party licence notice")
endif()
