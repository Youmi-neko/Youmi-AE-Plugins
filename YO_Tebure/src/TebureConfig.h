// Shared between Tebure.cpp, TeburePiPL.r and Tebure.rc (preprocessor only).
// The PiPL out-flags MUST equal what GlobalSetup sets; Tebure.cpp static_asserts this.
#ifndef TB_CONFIG_H
#define TB_CONFIG_H

// PiPL name stays ASCII (one string for every UI language)
#define TB_NAME "YO Tebure Shake"
#define TB_MATCH_NAME "CLD Tebure"
#define TB_CATEGORY "Youmi"
#define TB_SUPPORT_URL ""

#define TB_MAJOR_VERSION 0
#define TB_MINOR_VERSION 3
#define TB_BUG_VERSION 1
#define TB_BUILD_VERSION 1
// PF_VERSION(0, 3, 1, PF_Stage_DEVELOP, 1) = (3 << 15) | (1 << 11) | 1
#define TB_PIPL_VERSION 100353
#define TB_VERSION_STR "0.3.1.1"
#define TB_FILEVERSION 0, 3, 1, 1

// PF_OutFlag_WIDE_TIME_INPUT (1<<1: params are read at other times - speed / rhythm integration, impact keys)
// | PF_OutFlag_NON_PARAM_VARY (1<<2: the output changes over time with constant params)
// | PF_OutFlag_I_EXPAND_BUFFER (1<<9: "Transparent (Overflow)") | PF_OutFlag_CUSTOM_UI (1<<15: preset picker)
// | PF_OutFlag_DEEP_COLOR_AWARE (1<<25)
#define TB_OUTFLAGS 0x02008206
// PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG (1<<3) | PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10)
// | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12) | PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT (1<<17)
// | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27)
#define TB_OUTFLAGS2 0x08021408

#endif
