// Shared between MapBilinearBlur.cpp and MapBilinearBlurPiPL.r (preprocessor only).
// The PiPL out-flags MUST equal what GlobalSetup sets; MapBilinearBlur.cpp static_asserts this.
#ifndef MBB_CONFIG_H
#define MBB_CONFIG_H

// PiPL name stays ASCII (one string for every UI language).
#define MBB_NAME "YO Map Bilinear Blur"
#define MBB_MATCH_NAME "CLD MapBilinearBlur"
#define MBB_CATEGORY "Youmi"
#define MBB_SUPPORT_URL ""

#define MBB_MAJOR_VERSION 0
#define MBB_MINOR_VERSION 3
#define MBB_BUG_VERSION 1
#define MBB_BUILD_VERSION 1
// PF_VERSION(0, 3, 1, PF_Stage_DEVELOP, 1) = (3 << 15) | (1 << 11) | 1
#define MBB_PIPL_VERSION 100353
#define MBB_VERSION_STR "0.3.1.1"
#define MBB_FILEVERSION 0, 3, 1, 1

// PF_OutFlag_NON_PARAM_VARY (1<<2, cleared by QUERY_DYNAMIC_FLAGS unless animated dither)
// | PF_OutFlag_I_EXPAND_BUFFER (1<<9: "Spread Outside" grows the result) | PF_OutFlag_DEEP_COLOR_AWARE (1<<25)
#define MBB_OUTFLAGS 0x02000204
// PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS (1<<0) | PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10)
// | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12) | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27)
#define MBB_OUTFLAGS2 0x08001401

#endif
