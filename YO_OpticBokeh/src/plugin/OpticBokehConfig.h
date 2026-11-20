// Shared between OpticBokeh.cpp and OpticBokehPiPL.r (keep it preprocessor-only).
// The PiPL out-flags MUST equal what GlobalSetup sets; OpticBokeh.cpp static_asserts this.
#ifndef OPTICBOKEH_CONFIG_H
#define OPTICBOKEH_CONFIG_H

#ifdef OB_KANTAN
#define OB_NAME "YO Optic Bokeh Kantan"
#define OB_MATCH_NAME "CLD OpticBokehKantan"
#define OB_RC_DESCRIPTION "YO Optic Bokeh Kantan (simple edition) lens blur effect plug-in"
#define OB_RC_INTERNAL "YO_OpticBokehKantan"
#define OB_RC_FILENAME "YO_OpticBokehKantan.aex"
#else
#define OB_NAME "YO Optic Bokeh"
#define OB_MATCH_NAME "CLD OpticBokeh"
#define OB_RC_DESCRIPTION "YO Optic Bokeh lens blur effect plug-in"
#define OB_RC_INTERNAL "YO_OpticBokeh"
#define OB_RC_FILENAME "YO_OpticBokeh.aex"
#endif
#define OB_CATEGORY "Youmi"
// How Japanese parameter / popup names reach AE (obs::HostEncoding): 0 = English only,
// 1 = Windows ANSI code page (CP932), 2 = UTF-8, 3 = by AE version (UTF-8 on AE 26, ANSI otherwise).
#define OB_HOST_STRING_ENCODING 3
#define OB_SUPPORT_URL ""

#define OB_MAJOR_VERSION 2
#define OB_MINOR_VERSION 4
#define OB_BUG_VERSION 2
#define OB_BUILD_VERSION 1
// PF_VERSION(2, 4, 2, PF_Stage_RELEASE, 1) = (2 << 19) | (4 << 15) | (2 << 11) | (3 << 9) | 1
#define OB_STAGE_IS_RELEASE 1
#define OB_PIPL_VERSION 1185281
// VERSIONINFO (OpticBokeh.rc)
#define OB_VERSION_STR "2.4.2.1"
#define OB_FILEVERSION 2, 4, 2, 1

// PF_OutFlag_NON_PARAM_VARY (1<<2) | PF_OutFlag_I_EXPAND_BUFFER (1<<9) | PF_OutFlag_CUSTOM_UI (1<<15)
// | PF_OutFlag_DEEP_COLOR_AWARE (1<<25) | PF_OutFlag_SEND_UPDATE_PARAMS_UI (1<<26)
#define OB_OUTFLAGS 0x06008204
// PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS (1<<0) | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG (1<<3)
// | PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10) | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12)
// | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27)
#define OB_OUTFLAGS2 0x08001409

#endif
