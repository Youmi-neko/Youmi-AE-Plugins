# YO Optic Bokeh Kantan 2.4.2 - CMake build for Windows (x64 .aex) and macOS (universal arm64 + x86_64 .plugin).
#
# macOS:
#   cmake -S . -B build -G Ninja -DAE_SDK_DIR=/path/to/AfterEffectsSDK_26.5_mac/Examples -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
# Windows (MSVC 2022 x64 Native Tools prompt):
#   cmake -S . -B build -G Ninja -DAE_SDK_DIR=...\AfterEffectsSDK_26.5_win\Examples -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Picture rule: the engine must be compiled WITHOUT floating-point contraction and WITHOUT fast-math
# (MSVC: /fp:precise; clang: -ffp-contract=off). Do not add -ffast-math, -Ofast, /fp:fast, /arch:AVX2 or -mfma.
cmake_minimum_required(VERSION 3.21)

if(APPLE)
  if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "universal build" FORCE)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "first macOS with arm64" FORCE)
  endif()
endif()

project(OpticBokehKantan VERSION 2.4.2.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(AE_SDK_DIR "" CACHE PATH "After Effects SDK 'Examples' directory (contains Headers/, Resources/, Util/).")
set(OBGPU_LIB "" CACHE FILEPATH "Prebuilt obgpu static library (Windows obgpu.lib / macOS universal libobgpu.a). Empty: build with cargo.")
option(OB_SSE2_EMULATION "x64 only: compile the engine with the portable SSE2 stand-in used on arm64" OFF)
option(OB_STATIC_CRT "Windows: static CRT (/MT)" ON)
option(OB_WITH_GPU "Link the Rust/wgpu GPU stage (OFF: CPU-only plug-in, no cargo required)" ON)
option(OB_MAC_GPU_DEFAULT_ON "macOS: use the GPU without OPTICBOKEH_GPU=1" OFF)

set(OB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(OB_GPU ${CMAKE_CURRENT_SOURCE_DIR}/gpu)

if(MSVC AND OB_STATIC_CRT)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# ---------------------------------------------------------------- compiler flags (picture-relevant)
if(MSVC)
  add_compile_options(/utf-8 /EHsc /W3 /fp:precise)
else()
  add_compile_options(-ffp-contract=off -fno-fast-math -fno-builtin-pow -fno-builtin-powf -fno-strict-aliasing -fvisibility=hidden
                      $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>)
  if(APPLE)
    add_compile_options(-fmath-errno)
  endif()
  set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

set(OB_COMPAT_INC "")
if(APPLE OR (NOT MSVC AND (OB_SSE2_EMULATION OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM64)")))
  set(OB_COMPAT_INC ${OB_SRC}/platform/compat)
endif()
if(MSVC AND OB_SSE2_EMULATION)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FI${OB_SRC}/platform/ob_sse2_force_emul.h>")
endif()
set(OB_ENGINE_DEFS "")
if(OB_SSE2_EMULATION)
  list(APPEND OB_ENGINE_DEFS OB_SSE2_EMULATION)
endif()

function(ob_engine_tu tgt)
  if(OB_COMPAT_INC)
    target_include_directories(${tgt} BEFORE PRIVATE ${OB_COMPAT_INC})
  endif()
  target_compile_definitions(${tgt} PRIVATE ${OB_ENGINE_DEFS})
endfunction()

if(WIN32)
  set(OB_PLATFORM_SRC ${OB_SRC}/platform/ob_platform_win.cpp)
elseif(APPLE)
  set(OB_PLATFORM_SRC ${OB_SRC}/platform/ob_platform_mac.cpp)
else()
  message(FATAL_ERROR "only Windows and macOS are supported")
endif()

# ---------------------------------------------------------------- the engine (core/)
add_library(ob_engine OBJECT
  ${OB_SRC}/core/lenscore.cpp
  ${OB_SRC}/core/hq/lenshq.cpp
  ${OB_SRC}/core/hq/hq_model.cpp)
target_include_directories(ob_engine BEFORE PUBLIC ${OB_COMPAT_INC})
target_include_directories(ob_engine PUBLIC ${OB_SRC}/core ${OB_SRC}/core/hq)
target_compile_definitions(ob_engine PUBLIC ${OB_ENGINE_DEFS})
set_target_properties(ob_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ob_platform OBJECT ${OB_PLATFORM_SRC})
target_include_directories(ob_platform PUBLIC ${OB_SRC}/platform)
set_target_properties(ob_platform PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------- the GPU library (gpu/, Rust)
if(NOT OB_WITH_GPU)
  add_custom_target(obgpu_build)
  add_library(obgpu INTERFACE)
elseif(OBGPU_LIB)
  set(OB_GPU_LIBFILE ${OBGPU_LIB})
  add_custom_target(obgpu_build)
else()
  find_program(CARGO cargo HINTS $ENV{HOME}/.cargo/bin $ENV{USERPROFILE}/.cargo/bin REQUIRED)
  set(OB_CARGO_TARGET ${CMAKE_BINARY_DIR}/cargo)
  file(GLOB_RECURSE OB_GPU_DEPS CONFIGURE_DEPENDS ${OB_GPU}/src/* ${OB_GPU}/include/* ${OB_GPU}/cpp/*)
  list(APPEND OB_GPU_DEPS ${OB_GPU}/Cargo.toml ${OB_GPU}/Cargo.lock ${OB_GPU}/build.rs
       ${OB_SRC}/core/lenscore.cpp ${OB_SRC}/core/lenscore.h ${OB_SRC}/core/lens_presets.h)
  if(APPLE)
    set(OB_GPU_LIBFILE ${CMAKE_BINARY_DIR}/libobgpu.a)
    set(_cargo_cmds "")
    set(_slices "")
    foreach(arch IN LISTS CMAKE_OSX_ARCHITECTURES)
      if(arch STREQUAL "arm64")
        set(triple aarch64-apple-darwin)
      elseif(arch STREQUAL "x86_64")
        set(triple x86_64-apple-darwin)
      else()
        message(FATAL_ERROR "unsupported arch ${arch}")
      endif()
      list(APPEND _cargo_cmds COMMAND ${CMAKE_COMMAND} -E env MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
           ${CARGO} rustc --release --lib --crate-type staticlib --target ${triple} --target-dir ${OB_CARGO_TARGET}
           --manifest-path ${OB_GPU}/Cargo.toml)
      list(APPEND _slices ${OB_CARGO_TARGET}/${triple}/release/libobgpu.a)
    endforeach()
    add_custom_command(OUTPUT ${OB_GPU_LIBFILE}
      ${_cargo_cmds}
      COMMAND lipo -create ${_slices} -output ${OB_GPU_LIBFILE}
      DEPENDS ${OB_GPU_DEPS}
      COMMENT "cargo (wgpu / Metal) for ${CMAKE_OSX_ARCHITECTURES} + lipo"
      VERBATIM)
  else()
    set(OB_GPU_LIBFILE ${OB_CARGO_TARGET}/release/obgpu.lib)
    add_custom_command(OUTPUT ${OB_GPU_LIBFILE}
      COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-C target-feature=+crt-static"
              ${CARGO} rustc --release --lib --crate-type staticlib --target-dir ${OB_CARGO_TARGET}
              --manifest-path ${OB_GPU}/Cargo.toml
      DEPENDS ${OB_GPU_DEPS}
      COMMENT "cargo (wgpu / Vulkan + DX12, static CRT)"
      VERBATIM)
  endif()
  add_custom_target(obgpu_build DEPENDS ${OB_GPU_LIBFILE})
endif()

if(OB_WITH_GPU)
  add_library(obgpu INTERFACE)
  target_link_libraries(obgpu INTERFACE ${OB_GPU_LIBFILE})
  add_dependencies(obgpu obgpu_build)
  if(WIN32)
    target_link_libraries(obgpu INTERFACE kernel32 ntdll userenv ws2_32 dbghelp)
  elseif(APPLE)
    target_link_libraries(obgpu INTERFACE
      "-framework Metal" "-framework QuartzCore" "-framework CoreGraphics" "-framework Foundation"
      "-framework IOSurface" "-framework CoreFoundation" objc iconv)
  endif()
endif()

set(OB_PREPDESC_INC ${OB_SRC}/core/hq ${OB_SRC}/core ${OB_GPU}/include)

set(OB_UI_SRC ${OB_SRC}/plugin/ob_strings.cpp ${OB_SRC}/plugin/ob_thumbs.cpp ${OB_SRC}/plugin/ob_gpu.cpp ${OB_SRC}/plugin/kantan.cpp)
if(OB_WITH_GPU)
  list(APPEND OB_UI_SRC ${OB_GPU}/cpp/obgpu_prepdesc.cpp)
endif()
add_library(ob_ui OBJECT ${OB_UI_SRC})
ob_engine_tu(ob_ui)
target_include_directories(ob_ui PRIVATE ${OB_SRC}/plugin ${OB_PREPDESC_INC} ${OB_SRC}/core/hq)
if(NOT OB_WITH_GPU)
  target_compile_definitions(ob_ui PRIVATE OB_NO_GPU)
endif()
if(OB_MAC_GPU_DEFAULT_ON)
  target_compile_definitions(ob_ui PRIVATE OB_MAC_GPU_DEFAULT_ON=1)
endif()
set_target_properties(ob_ui PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(ob_ui obgpu_build)

# ---------------------------------------------------------------- After Effects SDK
if(NOT AE_SDK_DIR OR NOT EXISTS ${AE_SDK_DIR}/Headers/AE_Effect.h)
  message(FATAL_ERROR "Set -DAE_SDK_DIR=<path to AfterEffectsSDK/Examples> (must contain Headers/AE_Effect.h)")
endif()
set(OB_SDK_INC ${AE_SDK_DIR}/Headers ${AE_SDK_DIR}/Headers/SP ${AE_SDK_DIR}/Util)
if(WIN32)
  list(APPEND OB_SDK_INC ${AE_SDK_DIR}/Headers/Win)
endif()

set(OB_WRAPPER_INC ${OB_SRC}/plugin ${OB_SRC}/core ${OB_SRC}/core/hq ${OB_GPU}/include)
if(WIN32)
  set(OB_WRAPPER_DEFS MSWindows WIN32 _WINDOWS)
else()
  set(OB_WRAPPER_DEFS "")
endif()

# ---------------------------------------------------------------- the plug-in (YO_OpticBokehKantan)
set(tgt OpticBokehKantan)
set(outname YO_OpticBokehKantan)
set(dispname "YO Optic Bokeh Kantan")
set(pipl_dir ${CMAKE_BINARY_DIR}/pipl_kantan)
file(MAKE_DIRECTORY ${pipl_dir})

if(WIN32)
  add_custom_command(OUTPUT ${pipl_dir}/PiPL.rc
    COMMAND ${CMAKE_COMMAND} -DSDK=${AE_SDK_DIR} -DR=${OB_SRC}/plugin/OpticBokehPiPL.r -DDIR=${pipl_dir}
            -DKANTAN=1 -DCL=${CMAKE_CXX_COMPILER} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pipl_win.cmake
    DEPENDS ${OB_SRC}/plugin/OpticBokehPiPL.r ${OB_SRC}/plugin/OpticBokehConfig.h ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pipl_win.cmake
    COMMENT "PiPL (YO_OpticBokehKantan)" VERBATIM)
  add_library(${tgt} SHARED ${OB_SRC}/plugin/OpticBokeh.cpp ${pipl_dir}/PiPL.rc ${OB_SRC}/plugin/OpticBokeh.rc)
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${outname} SUFFIX ".aex" PREFIX "")
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  target_link_options(${tgt} PRIVATE /OPT:REF /OPT:ICF /MANIFEST:NO)
else()
  set(rsrc ${pipl_dir}/${outname}.rsrc)
  set(OB_REZ_SYSROOT "${CMAKE_OSX_SYSROOT}")
  if(NOT IS_DIRECTORY "${OB_REZ_SYSROOT}")
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE OB_REZ_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  set(rezsys "")
  if(IS_DIRECTORY "${OB_REZ_SYSROOT}")
    set(rezsys -isysroot "${OB_REZ_SYSROOT}")
  endif()
  set(rezdefs -d AE_OS_MAC -d __MACH__ -d __arm64__ -d OB_KANTAN)
  add_custom_command(OUTPUT ${rsrc}
    COMMAND xcrun Rez -o ${rsrc} -useDF -script Roman ${rezdefs}
            -i ${AE_SDK_DIR}/Headers -i ${AE_SDK_DIR}/Headers/SP -i ${AE_SDK_DIR}/Resources -i ${OB_SRC}/plugin
            ${rezsys} ${OB_SRC}/plugin/OpticBokehPiPL.r
    DEPENDS ${OB_SRC}/plugin/OpticBokehPiPL.r ${OB_SRC}/plugin/OpticBokehConfig.h
    COMMENT "Rez PiPL (YO_OpticBokehKantan)" VERBATIM)
  set(OB_TPL "")
  if(OB_WITH_GPU AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md ${pipl_dir}/THIRD_PARTY_NOTICES.md COPYONLY)
    set(OB_TPL ${pipl_dir}/THIRD_PARTY_NOTICES.md)
  endif()
  add_library(${tgt} MODULE ${OB_SRC}/plugin/OpticBokeh.cpp ${rsrc} ${OB_TPL})
  set_source_files_properties(${rsrc} ${OB_TPL} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  set(OB_BUNDLE_EXECUTABLE ${outname})
  set(OB_BUNDLE_ID "com.youmi.${tgt}")
  set(OB_BUNDLE_NAME "${dispname}")
  set(OB_BUNDLE_VERSION_SHORT "2.4.2")
  set(OB_BUNDLE_VERSION "2.4.2.1")
  configure_file(${OB_SRC}/plugin/mac/Info.plist.in ${pipl_dir}/Info.plist @ONLY)
  set_target_properties(${tgt} PROPERTIES
    BUNDLE TRUE BUNDLE_EXTENSION plugin
    MACOSX_BUNDLE_INFO_PLIST ${pipl_dir}/Info.plist
    OUTPUT_NAME ${outname})
  target_compile_options(${tgt} PRIVATE -Werror=return-type)
  target_link_options(${tgt} PRIVATE "LINKER:-exported_symbols_list,${OB_SRC}/plugin/mac/exports.txt" "LINKER:-dead_strip")
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DOUT=$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>/PkgInfo -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkginfo.cmake
    COMMENT "PkgInfo (eFKT / FXTC)" VERBATIM)
endif()

ob_engine_tu(${tgt})
target_include_directories(${tgt} PRIVATE ${OB_WRAPPER_INC} ${OB_SDK_INC})
target_compile_definitions(${tgt} PRIVATE ${OB_WRAPPER_DEFS} OB_KANTAN)
target_sources(${tgt} PRIVATE $<TARGET_OBJECTS:ob_engine> $<TARGET_OBJECTS:ob_ui> $<TARGET_OBJECTS:ob_platform>)
target_link_libraries(${tgt} PRIVATE obgpu)
if(APPLE)
  target_link_libraries(${tgt} PRIVATE "-framework CoreFoundation")
endif()
