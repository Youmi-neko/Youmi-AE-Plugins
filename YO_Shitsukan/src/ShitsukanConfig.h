// Shared between Shitsukan.cpp, ShitsukanPiPL.r and Shitsukan.rc (preprocessor only).
// The PiPL out-flags MUST equal what GlobalSetup sets; Shitsukan.cpp static_asserts this.
#ifndef SK_CONFIG_H
#define SK_CONFIG_H

// PiPL name stays ASCII (one string for every UI language)
#define SK_NAME "YO Shitsukan"
#define SK_MATCH_NAME "CLD Shitsukan"
#define SK_CATEGORY "Youmi"
#define SK_SUPPORT_URL ""

#define SK_MAJOR_VERSION 0
#define SK_MINOR_VERSION 6
#define SK_BUG_VERSION 0
#define SK_BUILD_VERSION 6
// PF_VERSION(0, 6, 0, PF_Stage_DEVELOP, 6) = (6 << 15) | 6
#define SK_PIPL_VERSION 196614
#define SK_VERSION_STR "0.6.0.6"
#define SK_FILEVERSION 0, 6, 0, 6

// PF_OutFlag_NON_PARAM_VARY (1<<2: only while "Move Paper" is on, QUERY_DYNAMIC_FLAGS clears it otherwise)
// | PF_OutFlag_DEEP_COLOR_AWARE (1<<25)
#define SK_OUTFLAGS 0x02000004
// PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS (1<<0) | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG (1<<3)
// | PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10) | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12)
// | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27)
#define SK_OUTFLAGS2 0x08001409

#endif
