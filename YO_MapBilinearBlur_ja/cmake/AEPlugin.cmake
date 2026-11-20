# AEPlugin.cmake - shared helper (identical in the Tebure / MapBilinearBlur / VectorNobi mac_port packages).
#
#   ae_add_effect(<target>
#       OUTPUT_NAME <base name>          # Tebure_proto -> Tebure_proto.plugin (mac) / Tebure_proto.aex (win)
#       PIPL <file.r>                    # PiPL resource source (must list CodeMacARM64 + CodeMacIntel64 + CodeWin64X86)
#       [RC <file.rc>]                   # Windows VERSIONINFO (ignored on macOS)
#       BUNDLE_ID <com.example.id>       # CFBundleIdentifier (macOS)
#       VERSION <x.y.z.w>                # CFBundleVersion / CFBundleShortVersionString
#       [PIPL_DEFINES <d1> <d2> ...]     # extra preprocessor defines for the PiPL (per-variant builds)
#       [PIPL_INCLUDES <dir> ...]        # extra include dirs for the PiPL (the source dir is always added)
#       SOURCES <...>)
#
# macOS : universal (CMAKE_OSX_ARCHITECTURES, set by the top-level CMakeLists before project()) MODULE bundle
#         <name>.plugin/Contents/{Info.plist, PkgInfo, MacOS/<name>, Resources/<name>.rsrc}; the PiPL is compiled
#         with `xcrun Rez` into the data fork of Resources/<name>.rsrc (what the SDK's Xcode projects do).
#         Only EffectMain and PluginDataEntryFunction2 are exported (everything else hidden).
# Windows: <name>.aex DLL, static CRT; PiPL via cl /EP -> PiPLtool.exe -> cl /EP -> rc (same chain as build.bat),
#         run by cmake/pipl_win.cmake.
#
# UNVERIFIED ON MAC: this file was written on Windows; the macOS branch has never been run.
include_guard(GLOBAL)

set(AE_SDK "" CACHE PATH "After Effects SDK 'Examples' folder (contains Headers/, Util/, Resources/)")

function(ae_require_sdk)
  if(NOT EXISTS "${AE_SDK}/Headers/AE_Effect.h")
    message(FATAL_ERROR "Set -DAE_SDK=<...>/AfterEffectsSDK_<ver>_mac/Examples (Headers/AE_Effect.h not found under '${AE_SDK}')")
  endif()
endfunction()

# SDK include paths + platform defines for a target that includes AE headers (plug-in or fake host)
function(ae_target_sdk tgt)
  target_include_directories(${tgt} PRIVATE "${AE_SDK}/Headers" "${AE_SDK}/Headers/SP" "${AE_SDK}/Util")
  if(WIN32)
    target_compile_definitions(${tgt} PRIVATE MSWindows WIN32 _WINDOWS)
  endif()
endfunction()

# Bit-exactness flags: no FP contraction (no FMA fusion), no fast-math, precise model, C++17.
function(ae_strict_fp tgt)
  # pinned, not a minimum: the u8"" string tables are const char* (C++17); a C++20 default (newer Xcode / -G Xcode)
  # would make them char8_t and stop the build
  target_compile_features(${tgt} PRIVATE cxx_std_17)
  set_target_properties(${tgt} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${tgt} PRIVATE /fp:precise /utf-8 /EHsc /W3)
  elseif(MSVC)  # clang-cl
    # clang-cl ignores bare -f options: pass them through /clang:
    target_compile_options(${tgt} PRIVATE /fp:precise /utf-8 /EHsc /clang:-ffp-contract=off /clang:-fno-fast-math)
  else()
    target_compile_options(${tgt} PRIVATE -ffp-contract=off -fno-fast-math -fno-finite-math-only -fno-unsafe-math-optimizations)
    if(APPLE)
      # Apple clang defaults to -fno-math-errno, which lets LLVM merge sin(x) + cos(x) into one __sincos_stret call (a
      # different libm entry point than the separate sin / cos the Windows build calls). -fmath-errno keeps them separate.
      # Checked on every slice by mac_common/mac_binary_gates.sh (run by tests/run_mac_tests.sh): no sincos import
      target_compile_options(${tgt} PRIVATE -fmath-errno)
    endif()
  endif()
  if(WIN32)
    set_property(TARGET ${tgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")  # static CRT like build.bat (/MT)
  endif()
endfunction()

# Compile one source file with AVX on x86_64 only (runtime dispatch decides whether it is called).
# On a universal macOS build the flag must be per-arch: -Xarch_x86_64 -mavx (arm64 compiles the file's #else branch).
function(ae_avx_source src)
  if(APPLE)
    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-Xarch_x86_64;-mavx")
  elseif(MSVC)
    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-mavx")
  endif()
endfunction()

function(ae_add_effect tgt)
  cmake_parse_arguments(A "" "OUTPUT_NAME;PIPL;RC;BUNDLE_ID;VERSION" "SOURCES;PIPL_DEFINES;PIPL_INCLUDES" ${ARGN})
  ae_require_sdk()
  get_filename_component(_pipl "${A_PIPL}" ABSOLUTE)
  get_filename_component(_pipldir "${_pipl}" DIRECTORY)
  # the PiPL also depends on the headers it includes (the plug-in's <Name>Config.h: name, match name, out flags,
  # version) and on the SDK's type declarations: list them so an incremental build never keeps a stale PiPL
  file(GLOB _piplhdrs CONFIGURE_DEPENDS "${_pipldir}/*.h")
  set(_pipldeps "${_pipl}" ${_piplhdrs} "${AE_SDK}/Headers/AEConfig.h" "${AE_SDK}/Headers/AE_EffectVers.h")
  if(EXISTS "${AE_SDK}/Resources/AE_General.r")
    list(APPEND _pipldeps "${AE_SDK}/Resources/AE_General.r")
  endif()
  add_library(${tgt} MODULE ${A_SOURCES})
  ae_target_sdk(${tgt})
  ae_strict_fp(${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME "${A_OUTPUT_NAME}" PREFIX "")

  if(APPLE)
    set(_rsrc "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT_NAME}.rsrc")
    set(_defs -d __MACH__)
    foreach(d IN LISTS A_PIPL_DEFINES)
      list(APPEND _defs -d ${d})
    endforeach()
    set(_incs -i "${AE_SDK}/Headers" -i "${AE_SDK}/Resources" -i "${AE_SDK}/Util" -i "${_pipldir}")
    foreach(i IN LISTS A_PIPL_INCLUDES)
      list(APPEND _incs -i "${i}")
    endforeach()
    # -useDF: resources in the data fork of the .rsrc file (the form AE reads from Contents/Resources).
    add_custom_command(OUTPUT "${_rsrc}"
      COMMAND xcrun Rez ${_defs} ${_incs} -useDF -o "${_rsrc}" "${_pipl}"
      DEPENDS ${_pipldeps} VERBATIM COMMENT "Rez ${A_OUTPUT_NAME} PiPL")
    target_sources(${tgt} PRIVATE "${_rsrc}")
    set_source_files_properties("${_rsrc}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources GENERATED TRUE)

    set(AE_PLIST_EXECUTABLE "${A_OUTPUT_NAME}")
    set(AE_PLIST_ID "${A_BUNDLE_ID}")
    set(AE_PLIST_VERSION "${A_VERSION}")
    set(_plist "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT_NAME}-Info.plist")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Info.plist.in" "${_plist}" @ONLY)
    set_target_properties(${tgt} PROPERTIES
      BUNDLE TRUE
      BUNDLE_EXTENSION plugin
      MACOSX_BUNDLE_INFO_PLIST "${_plist}"
      CXX_VISIBILITY_PRESET hidden
      C_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
      XCODE_ATTRIBUTE_WRAPPER_EXTENSION plugin)
    target_link_options(${tgt} PRIVATE
      "LINKER:-exported_symbol,_EffectMain"
      "LINKER:-exported_symbol,_PluginDataEntryFunction2"
      "LINKER:-dead_strip")
    target_link_libraries(${tgt} PRIVATE "-framework CoreFoundation")
    # The plist CMake writes for a BUNDLE comes from MACOSX_BUNDLE_INFO_PLIST; copy ours over it anyway (belt and
    # braces) and write PkgInfo (eFKT = effect plug-in type, FXTC = AE creator), as Xcode's GENERATE_PKGINFO_FILE does.
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${_plist}" "$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>/Info.plist"
      COMMAND /bin/sh -c "printf 'eFKTFXTC' > \"$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>/PkgInfo\""
      VERBATIM)
  elseif(WIN32)
    set_target_properties(${tgt} PROPERTIES SUFFIX ".aex")
    set(_pd "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_pipl")
    file(MAKE_DIRECTORY "${_pd}")
    # cl /EP writes to stdout: a CMake script (cmake/pipl_win.cmake) does the redirection, so paths with non-ASCII
    # characters survive (cmd /c mangles them under Ninja). `cl` must be on PATH (vcvars), also for clang-cl builds.
    find_program(AE_CL_EXE cl REQUIRED)
    set(_dd "")
    foreach(d IN LISTS A_PIPL_DEFINES)
      string(APPEND _dd "${d}|")
    endforeach()
    set(_ii "${AE_SDK}/Headers|${_pipldir}|")
    foreach(i IN LISTS A_PIPL_INCLUDES)
      string(APPEND _ii "${i}|")
    endforeach()
    add_custom_command(OUTPUT "${_pd}/PiPL.rc"
      COMMAND "${CMAKE_COMMAND}" "-DCL=${AE_CL_EXE}" "-DPIPLTOOL=${AE_SDK}/Resources/PiPLtool.exe" "-DSRC=${_pipl}"
              "-DOUTDIR=${_pd}" "-DINCS=${_ii}" "-DDEFS=${_dd}" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pipl_win.cmake"
      DEPENDS ${_pipldeps} VERBATIM COMMENT "PiPL ${A_OUTPUT_NAME}")
    target_sources(${tgt} PRIVATE "${_pd}/PiPL.rc")
    if(A_RC)
      target_sources(${tgt} PRIVATE "${A_RC}")
      set_property(SOURCE "${A_RC}" APPEND PROPERTY INCLUDE_DIRECTORIES "${_pipldir}")
    endif()
    target_link_libraries(${tgt} PRIVATE version)
    target_link_options(${tgt} PRIVATE /OPT:REF /OPT:ICF)
  endif()
endfunction()
