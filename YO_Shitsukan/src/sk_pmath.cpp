// mac_port: the portable float functions (see sk_pmath.h). Compiled in only with SK_PORTABLE_MATH=1 (the default
// off Windows). MUST be compiled without floating-point contraction (clang -ffp-contract=off, no fast-math).
#include "sk_pmath.h"

#if SK_PORTABLE_MATH
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "sk_pmath_model.h"

namespace {
// generated at build time from tables/pm_<fn>.bin by tools/pm_bin2c.cpp: kPmTab_<fn>[] (sorted (input bits, MSVC
// result bits) pairs) and kPmTabN_<fn> (the number of pairs)
#include "sk_pmath_tables.inc"

// an open-addressing hash of one table (built once, on first use; thread-safe static init)
struct Hash
{
    std::vector<uint64_t> slot;  // (key << 32) | value; key 0xFFFFFFFF (a NaN, never a key) = empty
    uint32_t mask = 0;
    int shift = 0;
    unsigned n = 0;
    Hash(const uint32_t* t, unsigned pairs)
    {
        n = pairs;
        unsigned cap = 16;
        int bits = 4;
        while (cap < 2u * pairs + 2u)
        {
            cap <<= 1;
            ++bits;
        }
        slot.assign(cap, ~0ull);
        mask = cap - 1;
        shift = 32 - bits;
        for (unsigned i = 0; i < pairs; ++i)
        {
            const uint32_t k = t[2 * i], v = t[2 * i + 1];
            uint32_t h = (k * 0x9E3779B1u) >> shift;
            while ((uint32_t)(slot[h] >> 32) != 0xFFFFFFFFu)
                h = (h + 1) & mask;
            slot[h] = ((uint64_t)k << 32) | v;
        }
    }
    // true + *out = the MSVC result when the input is in the table
    inline bool find(uint32_t k, uint32_t* out) const
    {
        uint32_t h = (k * 0x9E3779B1u) >> shift;
        for (;;)
        {
            const uint64_t s = slot[h];
            const uint32_t sk = (uint32_t)(s >> 32);
            if (sk == k)
            {
                *out = (uint32_t)s;
                return true;
            }
            if (sk == 0xFFFFFFFFu)
                return false;
            h = (h + 1) & mask;
        }
    }
};
const Hash& H(int fn)
{
    static const Hash h0(kPmTab_exp, kPmTabN_exp), h1(kPmTab_log, kPmTabN_log), h2(kPmTab_sin, kPmTabN_sin), h3(kPmTab_cos, kPmTabN_cos),
        h4(kPmTab_pow22, kPmTabN_pow22), h5(kPmTab_powig, kPmTabN_powig), h6(kPmTab_powthr, kPmTabN_powthr), h7(kPmTab_powhalo, kPmTabN_powhalo);
    const Hash* a[8] = {&h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7};
    return *a[fn];
}
inline float Fix(int fn, float x, float model)
{
    if (!pmm::InTableDomain(fn, x))
        return model;
    uint32_t v;
    return H(fn).find(pmm::FBits(x), &v) ? pmm::BitsF(v) : model;
}
const float kInvGamma = 1.f / 2.2f;  // (sk_core.cpp's constant)
}  // namespace

namespace pm {
float ExpF(float x)
{
    if (x != x)
        return x;
    return Fix(0, x, pmm::expModel(x));
}
float LogF(float x)
{
    if (x != x)
        return x;
    return Fix(1, x, pmm::logModel(x));
}
float SinF(float x)
{
    float m;
    if (!pmm::sinModel(x, m))
        return (float)std::sin((double)x);  // |x| > 1e6: not modelled (the engines never get there)
    return x != x ? m : Fix(2, x, m);
}
float CosF(float x)
{
    float m;
    if (!pmm::cosModel(x, m))
        return (float)std::cos((double)x);
    return x != x ? m : Fix(3, x, m);
}
float PowF(float x, float y)
{
    const float m = pmm::powModel(x, y);
    if (x != x || y != y)
        return m;
    if (y == 2.2f)
        return Fix(4, x, m);
    if (y == kInvGamma)
        return Fix(5, x, m);
    if (x == pmm::kPowBaseThr)  // the line threshold 0.3 * 0.1^sens: a function of the exponent alone
        return Fix(6, y, m);
    if (x == pmm::kPowBaseHalo)  // the halo 0.6 * 0.05^halo
        return Fix(7, y, m);
    return m;  // another base and exponent: the model alone (close to MSVC, not proven equal)
}
float Atan2F(float y, float x)
{
    return pmm::atan2Model(y, x);  // the model alone (close to MSVC, not proven equal)
}
int Portable() { return 1; }
int TableEntries(int fn) { return fn >= 0 && fn < 8 ? (int)H(fn).n : -1; }
}  // namespace pm

#else
namespace pm {
int Portable() { return 0; }
int TableEntries(int) { return 0; }
}  // namespace pm
#endif
