# YO Shitsukan 0.6.0.6 - macOS CMake build (for Windows x64, use build_win.ps1)
#
#   cmake -S . -B build -G Ninja -DAE_SDK=/path/to/AfterEffectsSDK_26.5_mac/Examples -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Produces build/YO_Shitsukan.plugin (universal arm64 + x86_64).
# GPU is OFF by default on macOS (-DSK_WITH_GPU=ON links gpu/ for Metal; opt-in via SHITSUKAN_GPU=1).
cmake_minimum_required(VERSION 3.21)
project(Shitsukan VERSION 0.6.0.6 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
  if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "" FORCE)
  endif()
endif()

set(AE_SDK "" CACHE PATH "After Effects SDK 'Examples' folder (contains Headers/, Resources/, Util/).")
option(SK_WITH_GPU "Link the Rust + wgpu GPU library (gpu/, Metal)" OFF)

# ---- floating-point rules: no FMA contraction, no fast-math, deterministic output
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  add_compile_options(-ffp-contract=off -fno-fast-math -fno-finite-math-only -fno-associative-math -fno-reciprocal-math
                      -fsigned-zeros -fno-builtin-pow)
  if(APPLE)
    add_compile_options(-fmath-errno)
  endif()
endif()
add_compile_definitions(NDEBUG SK_STR_ENC=3)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN})

# ---- portable math tables: tools/pm_bin2c converts src/tables/pm_*.bin into a C include
add_executable(pm_bin2c tools/pm_bin2c.cpp)
if(APPLE)
  set_target_properties(pm_bin2c PROPERTIES OSX_ARCHITECTURES "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
file(GLOB PM_TABLES ${SRC}/tables/pm_*.bin)
add_custom_command(OUTPUT ${GEN}/sk_pmath_tables.inc
  COMMAND pm_bin2c ${SRC}/tables ${GEN}/sk_pmath_tables.inc
  DEPENDS pm_bin2c ${PM_TABLES}
  COMMENT "pm_bin2c: portable math tables")
add_custom_target(pm_tables DEPENDS ${GEN}/sk_pmath_tables.inc)

# ---- engine
add_library(sk_engine STATIC ${SRC}/sk_core.cpp ${SRC}/sk_r6.cpp ${SRC}/sk_pmath.cpp)
add_dependencies(sk_engine pm_tables)
target_include_directories(sk_engine PUBLIC ${SRC} ${GEN})
set_target_properties(sk_engine PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
find_package(Threads REQUIRED)
target_link_libraries(sk_engine PUBLIC Threads::Threads)

# ---- GPU library (Rust, built by gpu/build_gpu_mac.sh into universal libskgpu.a)
if(SK_WITH_GPU)
  if(NOT APPLE)
    message(FATAL_ERROR "SK_WITH_GPU: this CMake file builds the GPU library for macOS only (Windows: build_win.ps1)")
  endif()
  file(GLOB GPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/src/* ${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/*)
  set(SKGPU_LIB ${CMAKE_CURRENT_BINARY_DIR}/gpu/libskgpu.a)
  add_custom_command(OUTPUT ${SKGPU_LIB}
    COMMAND /bin/sh ${CMAKE_CURRENT_SOURCE_DIR}/gpu/build_gpu_mac.sh ${CMAKE_CURRENT_BINARY_DIR}/gpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    DEPENDS ${GPU_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/Cargo.toml ${CMAKE_CURRENT_SOURCE_DIR}/gpu/Cargo.lock ${CMAKE_CURRENT_SOURCE_DIR}/gpu/build_gpu_mac.sh
    COMMENT "cargo: skgpu for aarch64 + x86_64 macOS, lipo")
  add_custom_target(skgpu DEPENDS ${SKGPU_LIB})
  add_dependencies(sk_engine skgpu)
  target_compile_definitions(sk_engine PUBLIC SK_GPU)
  target_include_directories(sk_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include)
  target_link_libraries(sk_engine PUBLIC ${SKGPU_LIB} "-framework Metal" "-framework QuartzCore" "-framework Foundation"
                        "-framework CoreGraphics" "-framework CoreFoundation" "-framework IOKit" objc iconv)
endif()

add_library(sk_strings STATIC ${SRC}/sk_strings.cpp)
target_include_directories(sk_strings PUBLIC ${SRC})
set_target_properties(sk_strings PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
if(APPLE)
  target_link_libraries(sk_strings PUBLIC "-framework CoreFoundation")
endif()

if(NOT AE_SDK OR NOT EXISTS "${AE_SDK}/Headers/AE_Effect.h")
  message(FATAL_ERROR "Set -DAE_SDK=<path to AfterEffectsSDK/Examples> (must contain Headers/AE_Effect.h)")
endif()
set(AE_INC ${AE_SDK}/Headers ${AE_SDK}/Headers/SP ${AE_SDK}/Util ${AE_SDK}/Resources)

# ---- plug-in bundle (macOS)
if(APPLE)
  set(PLUGIN_NAME Shitsukan)
  set(PLUGIN_OUTPUT_NAME YO_Shitsukan)
  add_library(${PLUGIN_NAME} MODULE ${SRC}/Shitsukan.cpp)
  target_include_directories(${PLUGIN_NAME} PRIVATE ${AE_INC} ${SRC})
  target_link_libraries(${PLUGIN_NAME} PRIVATE sk_engine sk_strings)
  set_target_properties(${PLUGIN_NAME} PROPERTIES
    OUTPUT_NAME "${PLUGIN_OUTPUT_NAME}"
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist.in
    MACOSX_BUNDLE_BUNDLE_NAME "YO Shitsukan"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.youmi.shitsukan"
    MACOSX_BUNDLE_BUNDLE_VERSION "0.6.0.6"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "0.6.0.6"
    XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN YES
    XCODE_ATTRIBUTE_ENABLE_HARDENED_RUNTIME YES)
  target_link_options(${PLUGIN_NAME} PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/mac/exports.txt" -Wl,-dead_strip)

  set(PKGINFO ${CMAKE_CURRENT_BINARY_DIR}/PkgInfo)
  file(WRITE ${PKGINFO} "eFKTFXTC")
  set(RSRC ${CMAKE_CURRENT_BINARY_DIR}/${PLUGIN_OUTPUT_NAME}.rsrc)
  add_custom_command(OUTPUT ${RSRC}
    COMMAND xcrun Rez -d __MACH__ -i ${AE_SDK}/Headers -i ${AE_SDK}/Headers/SP -i ${AE_SDK}/Resources -i ${AE_SDK}/Util -i ${SRC}
            -useDF -o ${RSRC} ${SRC}/ShitsukanPiPL.r
    DEPENDS ${SRC}/ShitsukanPiPL.r ${SRC}/ShitsukanConfig.h
    COMMENT "Rez: PiPL")
  add_custom_target(pipl DEPENDS ${RSRC})
  add_dependencies(${PLUGIN_NAME} pipl)
  add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:${PLUGIN_NAME}>/Resources
    COMMAND ${CMAKE_COMMAND} -E copy ${RSRC} $<TARGET_BUNDLE_CONTENT_DIR:${PLUGIN_NAME}>/Resources/${PLUGIN_OUTPUT_NAME}.rsrc
    COMMAND ${CMAKE_COMMAND} -E copy ${PKGINFO} $<TARGET_BUNDLE_CONTENT_DIR:${PLUGIN_NAME}>/PkgInfo
    COMMENT "bundle: PiPL .rsrc + PkgInfo")
  if(SK_WITH_GPU AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md)
    add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md $<TARGET_BUNDLE_CONTENT_DIR:${PLUGIN_NAME}>/Resources/THIRD_PARTY_NOTICES.md
      COMMENT "bundle: third-party licence notices")
  endif()
endif()
