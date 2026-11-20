// mac_port: Rez defines no processor macro, and AEConfig.h stops with "unrecognized AE processor" without one.
// The Mac code entries below list both architectures regardless, so any value works here. Windows (cl /EP) never
// defines Rez: the Windows PiPL is unchanged.
#if defined(Rez) && !defined(__x86_64__) && !defined(__arm64__) && !defined(__aarch64__) && !defined(__i386__)
#define __arm64__ 1
#endif
#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "MapBilinearBlurConfig.h"

#ifndef AE_OS_WIN
	#include "AE_General.r"
#endif

resource 'PiPL' (16000) {
	{
		Kind {
			AEEffect
		},
		Name {
			MBB_NAME
		},
		Category {
			MBB_CATEGORY
		},
#ifdef AE_OS_WIN
    #if defined(AE_PROC_INTELx64)
		CodeWin64X86 {"EffectMain"},
    #elif defined(AE_PROC_ARM64)
		CodeWinARM64 {"EffectMain"},
    #endif
#elif defined(AE_OS_MAC)
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		AE_PiPL_Version {
			2,
			0
		},
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		AE_Effect_Version {
			MBB_PIPL_VERSION
		},
		AE_Effect_Info_Flags {
			0
		},
		AE_Effect_Global_OutFlags {
			MBB_OUTFLAGS
		},
		AE_Effect_Global_OutFlags_2 {
			MBB_OUTFLAGS2
		},
		AE_Effect_Match_Name {
			MBB_MATCH_NAME
		},
		AE_Reserved_Info {
			8
		}
	}
};
