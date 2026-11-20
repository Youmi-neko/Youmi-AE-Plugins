// Tebure - presets (see tb_presets.h).
#include "tb_presets.h"
#include "platform/plat_math.h"  // tbm:: = the libm calls that feed the picture (forward to std::)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace tb {

namespace {
Shape P_Still()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.11, s.handHz = 0.9, s.handQ = 0.7, s.handRough = 0.35, s.handYX = 0.9, s.handRoll = 0.45;
    s.bodyPx = 1.5, s.tremorDeg = 0.006, s.breathDeg = 0.05, s.breathHz = 0.25, s.driftDeg = 0.12, s.corrInt = 3.0;
    s.swell = 0.3, s.handIrr = 0.6;
    return s;
}
Shape P_Tele()
{
    Shape s;
    s.focal = 200;
    s.handDeg = 0.045, s.handHz = 1.5, s.handQ = 0.8, s.handRough = 0.6, s.handYX = 0.9, s.handRoll = 0.3;
    s.bodyPx = 0.5, s.tremorDeg = 0.01, s.breathDeg = 0.025, s.breathHz = 0.27, s.driftDeg = 0.07, s.corrInt = 2.0;
    s.swell = 0.35, s.handIrr = 0.6;
    return s;
}
Shape P_Wide()
{
    Shape s;
    s.focal = 24;
    s.handDeg = 0.22, s.handHz = 0.65, s.handQ = 0.65, s.handRough = 0.25, s.handYX = 1.0, s.handRoll = 0.8;
    s.bodyPx = 5, s.tremorDeg = 0.004, s.breathDeg = 0.09, s.breathHz = 0.23, s.driftDeg = 0.3, s.corrInt = 4.0;
    s.swell = 0.35, s.handIrr = 0.6;
    return s;
}
Shape P_Walk()
{
    Shape s;
    s.focal = 28;
    s.handDeg = 0.1, s.handHz = 1.2, s.handQ = 0.7, s.handRough = 0.3, s.handYX = 0.9, s.handRoll = 0.4;
    s.bodyPx = 2, s.tremorDeg = 0.004, s.driftDeg = 0.2, s.corrInt = 2.5;
    s.cadence = 1.85, s.bouncePx = 9, s.swayPx = 6, s.walkRoll = 0.6, s.walkPan = 0.3, s.heelPx = 3, s.heelSharp = 0.15;
    s.gaitVar = 0.7, s.handIrr = 0.5;
    s.swell = 0.2;
    return s;
}
Shape P_Run()
{
    Shape s;
    s.focal = 24;
    s.handDeg = 0.22, s.handHz = 1.5, s.handQ = 0.7, s.handRough = 0.5, s.handYX = 1.0, s.handRoll = 0.5;
    s.bodyPx = 4, s.driftDeg = 0.3, s.corrInt = 1.5;
    s.cadence = 2.8, s.bouncePx = 20, s.swayPx = 9, s.walkRoll = 1.3, s.walkPan = 0.7, s.heelPx = 10, s.heelSharp = 0.8;
    s.gaitVar = 0.8, s.handIrr = 0.5;
    s.swell = 0.3;
    return s;
}
Shape P_Car()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.05, s.handHz = 1.0, s.handQ = 0.7, s.handRough = 0.3, s.handRoll = 0.4;
    s.vibPx = 1.1, s.vibHz = 28, s.vibBand = 0.15, s.bumpPx = 7, s.bumpRate = 0.6;
    s.bigPx = 4, s.bigHz = 0.6, s.bigRot = 0.25, s.swell = 0.45;
    s.shutter = 180;  // 28 Hz vibration reads as a car only with blur
    return s;
}
Shape P_Impact()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.03, s.handHz = 1.0, s.handRough = 0.3;
    s.impType = IMP_PHYSICAL, s.impPx = 30, s.impRot = 0.8, s.impZoom = 0.6, s.impHz = 5, s.impGrit = 0.2;
    s.impAngle = 90, s.impRandom = 0.35, s.decay = 0.35, s.firstHit = 0.25;
    return s;
}
Shape P_Explosion()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.03, s.handHz = 1.0, s.handRough = 0.3;
    s.impType = IMP_PHYSICAL, s.impPx = 45, s.impRot = 1.2, s.impZoom = 1.0, s.impHz = 3.5, s.impGrit = 0.6;
    s.impAngle = 90, s.impRandom = 0.4, s.decay = 0.9, s.firstHit = 0.25;
    s.vibPx = 1.5, s.vibHz = 11, s.vibBand = 0.5, s.swell = 0.5;
    s.shutter = 180;
    return s;
}
Shape P_Quake()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.04, s.handHz = 1.0, s.handRough = 0.3;
    s.bigPx = 34, s.bigHz = 0.7, s.bigRot = 0.7, s.bigZoom = 0.3;
    s.vibPx = 4, s.vibHz = 9, s.vibBand = 0.6, s.swell = 0.5;
    // P waves (rattle) first, the S wave (large slow motion) 1.8 s later, decaying over ~6 s
    s.quakeEnv = 1, s.firstHit = 0.5, s.decay = 6;
    return s;
}
Shape P_Tripod()
{
    Shape s;
    s.focal = 50;
    s.handDeg = 0.018, s.handHz = 4.5, s.handQ = 8, s.handRough = 0, s.handYX = 0.7, s.handRoll = 0.3;
    s.bigPx = 1.0, s.bigHz = 0.3, s.swell = 0.9;
    return s;
}
Shape P_Drone()
{
    Shape s;
    s.focal = 24;
    s.handDeg = 0.012, s.handHz = 0.6, s.handQ = 0.5, s.handRough = 0, s.handRoll = 0.5;
    s.bigPx = 12, s.bigHz = 0.08, s.bigRot = 0.25, s.bigZoom = 0.3, s.swell = 0.3;
    return s;
}
Shape P_Film()
{
    Shape s;
    s.focal = 35;
    s.frRate = FR_24;
    s.weavePx = 1.5, s.jitterPx = 0.8, s.jitterYX = 2.0;
    return s;
}
Shape P_Drunk()
{
    Shape s;
    s.focal = 28;
    s.handDeg = 0.1, s.handHz = 0.5, s.handQ = 0.6, s.handRough = 0.1, s.handRoll = 0.6, s.handIrr = 0.8;
    // slow roll and sway, lagging recentring (drift + late corrections), now and then a lurch
    s.bigPx = 14, s.bigHz = 0.1, s.bigRot = 1.8, s.bigZoom = 1.2, s.swell = 0.4;
    s.driftDeg = 0.45, s.corrInt = 5;
    s.joltPx = 22, s.joltInt = 7, s.joltDur = 0.6, s.joltSettle = 0.3, s.joltMode = JOLT_FREE, s.joltYX = 0.6;
    return s;
}
Shape P_Heart()
{
    Shape s;
    s.focal = 35;
    s.handDeg = 0.03, s.handHz = 0.9, s.handRough = 0.3;
    s.cadence = 1.2, s.pulsePct = 0.8;
    return s;
}
Shape P_GamenV()
{
    Shape s;
    s.focal = 35;
    s.frRate = FR_HOLD, s.holdN = 2;
    s.impType = IMP_GAMENDO, s.impPx = 8, s.impAngle = 90, s.impRandom = 0.15, s.decay = 0.6, s.firstHit = 0.25;
    return s;
}
Shape P_GamenD()
{
    Shape s;
    s.focal = 35;
    s.frRate = FR_HOLD, s.holdN = 2;
    s.impType = IMP_GAMENDO, s.impPx = 10, s.impAngle = 35, s.impRandom = 0.2, s.decay = 0.7, s.firstHit = 0.25;
    return s;
}
Shape P_AnimeHand()
{
    Shape s;
    s.focal = 35;
    s.frRate = FR_HOLD, s.holdN = 3;
    s.handDeg = 0.2, s.handHz = 0.9, s.handQ = 0.7, s.handRough = 0.3, s.handYX = 0.9, s.handRoll = 0.3;
    s.bodyPx = 4, s.swell = 0.3, s.handIrr = 0.6;
    return s;
}
struct Table
{
    Preset p[17];
    Table()
    {
        int i = 0;
        p[i++] = {"still", P_Still(), false};
        p[i++] = {"tele", P_Tele(), false};
        p[i++] = {"wide", P_Wide(), false};
        p[i++] = {"walk", P_Walk(), false};
        p[i++] = {"run", P_Run(), false};
        p[i++] = {"car", P_Car(), false};
        p[i++] = {"impact", P_Impact(), false};
        p[i++] = {"explosion", P_Explosion(), false};
        p[i++] = {"quake", P_Quake(), false};
        p[i++] = {"tripod", P_Tripod(), false};
        p[i++] = {"drone", P_Drone(), false};
        p[i++] = {"film", P_Film(), false};
        p[i++] = {"drunk", P_Drunk(), false};
        p[i++] = {"heart", P_Heart(), false};
        p[i++] = {"gamen_v", P_GamenV(), false};
        p[i++] = {"gamen_d", P_GamenD(), false};
        p[i++] = {"anime_hand", P_AnimeHand(), false};
    }
};
const Table& T()
{
    static const Table t;
    return t;
}
}  // namespace

const Preset* Presets(int* count)
{
    if (count)
        *count = 17;
    return T().p;
}

const FieldD* FieldsD(int* count)
{
    static const FieldD f[] = {
        {"focal", &Shape::focal}, {"handDeg", &Shape::handDeg}, {"handHz", &Shape::handHz}, {"handQ", &Shape::handQ},
        {"handRough", &Shape::handRough}, {"handYHz", &Shape::handYHz}, {"handIrr", &Shape::handIrr}, {"gaitVar", &Shape::gaitVar}, {"quakeEnv", &Shape::quakeEnv}, {"hitEvery", &Shape::hitEvery}, {"shutter", &Shape::shutter}, {"handYX", &Shape::handYX}, {"handRoll", &Shape::handRoll},
        {"bodyPx", &Shape::bodyPx}, {"bodyHz", &Shape::bodyHz}, {"tremorDeg", &Shape::tremorDeg}, {"breathDeg", &Shape::breathDeg},
        {"breathHz", &Shape::breathHz}, {"driftDeg", &Shape::driftDeg}, {"corrInt", &Shape::corrInt}, {"cadence", &Shape::cadence},
        {"bouncePx", &Shape::bouncePx}, {"swayPx", &Shape::swayPx}, {"walkRoll", &Shape::walkRoll}, {"walkPan", &Shape::walkPan},
        {"heelPx", &Shape::heelPx}, {"heelSharp", &Shape::heelSharp}, {"pulsePct", &Shape::pulsePct}, {"vibPx", &Shape::vibPx},
        {"vibHz", &Shape::vibHz}, {"vibBand", &Shape::vibBand}, {"bumpPx", &Shape::bumpPx}, {"bumpRate", &Shape::bumpRate},
        {"bigPx", &Shape::bigPx}, {"bigHz", &Shape::bigHz}, {"bigRot", &Shape::bigRot}, {"bigZoom", &Shape::bigZoom},
        {"swell", &Shape::swell}, {"impPx", &Shape::impPx}, {"impRot", &Shape::impRot}, {"impZoom", &Shape::impZoom},
        {"impHz", &Shape::impHz}, {"impGrit", &Shape::impGrit}, {"impAngle", &Shape::impAngle}, {"impRandom", &Shape::impRandom},
        {"decay", &Shape::decay}, {"firstHit", &Shape::firstHit}, {"joltPx", &Shape::joltPx}, {"joltInt", &Shape::joltInt},
        {"joltDur", &Shape::joltDur}, {"joltSettle", &Shape::joltSettle}, {"joltYX", &Shape::joltYX}, {"weavePx", &Shape::weavePx},
        {"jitterPx", &Shape::jitterPx}, {"jitterYX", &Shape::jitterYX}};
    if (count)
        *count = (int)(sizeof(f) / sizeof(f[0]));
    return f;
}
const FieldI* FieldsI(int* count)
{
    static const FieldI f[] = {{"frRate", &Shape::frRate}, {"holdN", &Shape::holdN}, {"impType", &Shape::impType}, {"joltMode", &Shape::joltMode}};
    if (count)
        *count = (int)(sizeof(f) / sizeof(f[0]));
    return f;
}
bool ApplyOverrides(Shape& s, const char* spec)
{
    if (!spec)
        return true;
    std::string all(spec);
    size_t st = 0;
    while (st < all.size())
    {
        size_t e = all.find(',', st);
        if (e == std::string::npos)
            e = all.size();
        const std::string kv = all.substr(st, e - st);
        st = e + 1;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos)
            continue;
        const std::string k = kv.substr(0, eq);
        const double v = std::atof(kv.c_str() + eq + 1);
        bool ok = false;
        int n = 0;
        const FieldD* fd = FieldsD(&n);
        for (int i = 0; i < n; ++i)
            if (k == fd[i].key)
                s.*(fd[i].m) = v, ok = true;
        const FieldI* fi = FieldsI(&n);
        for (int i = 0; i < n; ++i)
            if (k == fi[i].key)
                s.*(fi[i].m) = (int)v, ok = true;
        if (!ok)
            return false;
    }
    return true;
}

int PresetIndex(const char* key)
{
    int n = 0;
    const Preset* p = Presets(&n);
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(p[i].key, key))
            return i;
    return -1;
}

int DefaultPreset() { return PresetIndex("still"); }

int AutoPick(uint32_t seed, int64_t cut, double* amountFactor, double mood)
{
    static const char* keys[] = {"still", "tripod", "wide", "drone", "film", "walk", "anime_hand"};
    static const double w0[] = {0.35, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07};
    static const double live[] = {-0.6, -1.0, 0.2, -0.3, 0.0, 0.8, 1.0};
    mood = std::min(1.0, std::max(-1.0, mood));
    double w[7], sum = 0;
    for (int i = 0; i < 7; ++i)
        sum += (w[i] = w0[i] * tbm::exp(1.6 * mood * live[i]));
    const double u = Uni(Mix64(0xA070ull + seed), 1, cut) * sum;
    double acc = 0;
    int pick = 0;
    for (int i = 0; i < 7; ++i)
    {
        acc += w[i];
        if (u < acc)
        {
            pick = i;
            break;
        }
        pick = i;
    }
    if (amountFactor)
        *amountFactor = (0.75 + 0.55 * Uni(Mix64(0xA070ull + seed), 2, cut)) * (1 + 0.3 * mood);
    return PresetIndex(keys[pick]);
}

}  // namespace tb
