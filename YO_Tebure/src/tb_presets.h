// Tebure - preset table.
// Units: deg (camera rotation), px of a 1920 px wide frame (frame relative), %, Hz, s.
// The popup value v = index + 2 (1 = Custom).
#pragma once
#include "tb_model.h"

#include <cstring>

namespace tb {

struct Preset
{
    const char* key;
    Shape s;
    bool autoPerCut;
};

inline Shape MakeShape(void (*f)(Shape&))
{
    Shape s;
    f(s);
    return s;
}

// preset popup items (English; the Japanese table maps the whole string)
#define TB_PRESET_ITEMS                                                                                                       \
    "Custom|Handheld (Still)|Handheld (Tele)|Handheld (Wide)|Walking|Running|In a Car|Impact (Once)|Explosion / Rumble|"     \
    "Earthquake|Tripod in Wind|Drone Float|Film Gate Weave|Drunk / Dizzy|Heartbeat|Gamen-do (Vertical)|Gamen-do (Diagonal)|" \
    "Anime Handheld on 3s"

// reflective list of the shape fields (CLI overrides, the plug-in's control table cross-check, tests)
struct FieldD
{
    const char* key;
    double Shape::*m;
};
struct FieldI
{
    const char* key;
    int Shape::*m;
};
const FieldD* FieldsD(int* count);
const FieldI* FieldsI(int* count);
// "k=v,k=v" -> shape; false on an unknown key
bool ApplyOverrides(Shape& s, const char* spec);

const Preset* Presets(int* count);
int PresetIndex(const char* key);  // -1 if unknown
int DefaultPreset();               // index (popup value = index + 2)
int AutoPick(uint32_t seed, int64_t cut, double* amountFactor, double mood = 0);

}  // namespace tb
