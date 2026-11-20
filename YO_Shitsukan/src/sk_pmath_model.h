// Portable float math MODELS (mac_port). Deterministic on every IEEE-754 platform: only double +, -, *, /, floor,
// frexp, ldexp and conversions, evaluated in the written order - the translation unit MUST be compiled without
// floating-point contraction (clang: -ffp-contract=off, no -ffast-math; MSVC x64 /fp:precise without /arch:AVX2).
// The models are close to correctly rounded but are NOT the MSVC runtime: sk_pmath.cpp adds, per function, a table
// of the float inputs where the MSVC runtime (VS 2022, /O2 /MT, the CRT the Windows .aex links) returns another
// float (tools/gen_pmath_tables.cpp writes them on Windows by trying every float). Model + table = the MSVC result
// bit for bit, on any platform.
//
// expModel is EXACTLY tools/gen_exp_table.cpp's ExpModel and gpu/shaders/tail.wgsl's exp_ms (the same operations in
// the same order), so the GPU's exp_tab.bin and the CPU agree by construction.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace pmm {

inline double D(uint64_t b)
{
    double d;
    std::memcpy(&d, &b, 8);
    return d;
}
// (unions, not memcpy: MSVC 2022 /O2 miscompiled the memcpy version inside a loop over a truncated 64-bit counter -
// it converted the integer's VALUE to float. Union punning is defined by MSVC, GCC and clang.)
inline uint32_t FBits(float f)
{
    union
    {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    return v.u;
}
inline float BitsF(uint32_t u)
{
    union
    {
        uint32_t u;
        float f;
    } v;
    v.u = u;
    return v.f;
}

struct Consts
{
    double C[14];     // 1 / n!, n = 0..13 (exp)
    double P2[321];   // 2^k, k = -160..160
    double S[22];     // 1 / n!, n = 0..21 (sin / cos)
    double L[12];     // 1 / (2k + 1), k = 0..11 (log: atanh series)
    double A[14];     // (-1)^k / (2k + 1), k = 0..13 (atan series)
    Consts()
    {
        double f = 1.0;
        for (int n = 0; n < 14; ++n)
        {
            if (n > 0)
                f *= n;
            C[n] = 1.0 / f;
        }
        for (int k = -160; k <= 160; ++k)
            P2[k + 160] = std::ldexp(1.0, k);
        f = 1.0;
        for (int n = 0; n < 22; ++n)
        {
            if (n > 0)
                f *= n;  // (21! > 2^53: f is the rounded factorial; still deterministic)
            S[n] = 1.0 / f;
        }
        for (int k = 0; k < 12; ++k)
            L[k] = 1.0 / (2 * k + 1);
        for (int k = 0; k < 14; ++k)
            A[k] = (k & 1 ? -1.0 : 1.0) / (2 * k + 1);
    }
};
inline const Consts& K()
{
    static const Consts k;
    return k;
}
const double kInvLn2 = D(0x3FF71547652B82FEull), kLn2Hi = D(0x3FE62E42FEE00000ull), kLn2Lo = D(0x3DEA39EF35793C76ull);

// ---- exp(float): tools/gen_exp_table.cpp ExpModel (valid for |x| <= ~110; outside: +inf / +0)
inline float expModel(float x)
{
    if (x != x)
        return x + x;
    if (x > 110.f)
        return BitsF(0x7F800000u);
    if (x < -110.f)
        return 0.f;
    const Consts& k = K();
    const double xd = x;
    const double kd = std::floor(xd * kInvLn2 + 0.5);
    const double r = (xd - kd * kLn2Hi) - kd * kLn2Lo;
    double p = k.C[13];
    for (int n = 12; n >= 0; --n)
        p = p * r + k.C[n];
    return (float)(p * k.P2[(int)kd + 160]);
}

// ---- double helpers (results NOT rounded to float)
inline double expD(double t)  // |t| < ~700
{
    const Consts& k = K();
    if (t != t)
        return t;
    if (t > 709.0)
        return HUGE_VAL;
    if (t < -745.0)
        return 0.0;
    const double kd = std::floor(t * kInvLn2 + 0.5);
    const double r = (t - kd * kLn2Hi) - kd * kLn2Lo;
    double p = k.C[13];
    for (int n = 12; n >= 0; --n)
        p = p * r + k.C[n];
    return std::ldexp(p, (int)kd);
}
inline double logD(double x)  // x > 0 finite
{
    const Consts& k = K();
    int e = 0;
    double m = std::frexp(x, &e);  // [0.5, 1)
    if (m < 0.70710678118654752440)
    {
        m = m * 2.0;
        e -= 1;
    }
    const double s = (m - 1.0) / (m + 1.0);
    const double s2 = s * s;
    double p = k.L[11];
    for (int j = 10; j >= 0; --j)
        p = p * s2 + k.L[j];
    const double lm = (2.0 * s) * p;
    return (double)e * kLn2Hi + ((double)e * kLn2Lo + lm);
}

// ---- log(float)
inline float logModel(float x)
{
    if (x != x)
        return x + x;
    if (x == 0.f)
        return -BitsF(0x7F800000u);
    if (x < 0.f)
        return BitsF(0xFFC00000u);  // -nan(ind), as the MSVC runtime
    if (x == BitsF(0x7F800000u))
        return x;
    return (float)logD((double)x);
}

// ---- sin / cos(float): Cody-Waite reduction by pi/2 in three parts (exact products for |k| < 2^20)
const double kTwoOverPi = D(0x3FE45F306DC9C883ull);
const double kPio2_1 = D(0x3FF921FB54400000ull), kPio2_2 = D(0x3DD0B4611A600000ull), kPio2_3 = D(0x3BA3198A2E037073ull);
const double kTrigMax = 1.0e6;  // larger |x|: not modelled (sk_pmath.cpp falls back to the platform libm)
inline void sinCosR(double r, double& s, double& c)
{
    const Consts& k = K();
    const double r2 = r * r;
    double ps = k.S[21];
    for (int n = 19; n >= 1; n -= 2)
        ps = ps * r2 + (((n >> 1) & 1) ? -k.S[n] : k.S[n]);
    s = r * ps;
    double pc = k.S[20];
    for (int n = 18; n >= 0; n -= 2)
        pc = pc * r2 + (((n >> 1) & 1) ? -k.S[n] : k.S[n]);
    c = pc;
}
inline bool trigReduce(float x, double& r, int& q)
{
    const double xd = x;
    if (!(std::fabs(xd) <= kTrigMax))
        return false;
    const double kf = std::floor(xd * kTwoOverPi + 0.5);
    r = ((xd - kf * kPio2_1) - kf * kPio2_2) - kf * kPio2_3;
    q = (int)((long long)kf & 3);
    return true;
}
inline bool sinModel(float x, float& out)
{
    if (x != x)
    {
        out = x + x;
        return true;
    }
    double r;
    int q;
    if (!trigReduce(x, r, q))
        return false;
    if (x == 0.f)
    {
        out = x;
        return true;
    }
    double s, c;
    sinCosR(r, s, c);
    const double v = q == 0 ? s : q == 1 ? c : q == 2 ? -s : -c;
    out = (float)v;
    return true;
}
inline bool cosModel(float x, float& out)
{
    if (x != x)
    {
        out = x + x;
        return true;
    }
    double r;
    int q;
    if (!trigReduce(x, r, q))
        return false;
    double s, c;
    sinCosR(r, s, c);
    const double v = q == 0 ? c : q == 1 ? -s : q == 2 ? -c : s;
    out = (float)v;
    return true;
}

// ---- pow(float, float) as exp(y * log(x)) in double. Special cases as C99 for the cases the engines can produce
// (x > 0; x == 0; y == 0; x == 1); a negative x gives NaN unless y is an integer.
inline float powModel(float x, float y)
{
    if (y == 0.f || x == 1.f)
        return 1.f;
    if (x != x || y != y)
        return x + y;
    const double yd = y;
    if (x == 0.f)
        return y > 0.f ? 0.f : BitsF(0x7F800000u);
    if (x < 0.f)
    {
        if (std::floor(yd) != yd)
            return BitsF(0xFFC00000u);
        const double m = expD(yd * logD(-(double)x));
        const bool odd = std::fabs(std::fmod(yd, 2.0)) == 1.0;
        return (float)(odd ? -m : m);
    }
    if (x == BitsF(0x7F800000u))
        return y > 0.f ? x : 0.f;
    return (float)expD(yd * logD((double)x));
}

// ---- atan2(float, float) in double: |t| reduced to [0, 1], then twice halved (atan t = 2 atan(t / (1 + sqrt(1 + t^2))))
// to |t| <= tan(pi/16), then the series
const double kPi = D(0x400921FB54442D18ull), kPio2 = D(0x3FF921FB54442D18ull), kPio4 = D(0x3FE921FB54442D18ull);
inline double atanUnit(double t)  // 0 <= t <= 1
{
    const Consts& k = K();
    double u = t / (1.0 + std::sqrt(1.0 + t * t));
    u = u / (1.0 + std::sqrt(1.0 + u * u));
    const double u2 = u * u;
    double p = k.A[13];
    for (int j = 12; j >= 0; --j)
        p = p * u2 + k.A[j];
    return 4.0 * (u * p);
}
inline float atan2Model(float y, float x)
{
    if (x != x || y != y)
        return x + y;
    const double yd = y, xd = x;
    const bool yneg = std::signbit(y), xneg = std::signbit(x);
    const double ay = std::fabs(yd), axd = std::fabs(xd);
    const double inf = HUGE_VAL;
    double a;
    if (ay == 0.0)
        a = xneg ? kPi : 0.0;
    else if (axd == 0.0)
        a = kPio2;
    else if (ay == inf && axd == inf)
        a = xneg ? 3.0 * kPio4 : kPio4;
    else if (ay == inf)
        a = kPio2;
    else if (axd == inf)
        a = xneg ? kPi : 0.0;
    else
    {
        const double base = ay <= axd ? atanUnit(ay / axd) : kPio2 - atanUnit(axd / ay);
        a = xneg ? kPi - base : base;
    }
    return (float)(yneg ? -a : a);
}

// ---- the exception tables' domains (tools/gen_pmath_tables.cpp tries every float inside). fn: 0 exp, 1 log, 2 sin,
// 3 cos, 4 pow(x, 2.2f), 5 pow(x, 1.f / 2.2f), 6 pow(0.03f / 0.30f, y), 7 pow(0.03f / 0.6f, y) (6 / 7: the argument is
// the EXPONENT y: the line threshold 0.3 * 0.1^sens and the halo 0.6 * 0.05^halo, sens / halo in [0, 1]).
// Outside: the model alone (portable, not proven equal to MSVC).
inline bool InTableDomain(int fn, float x)
{
    const float ax = x < 0.f ? -x : x;
    switch (fn)
    {
    case 0: return true;                            // every float
    case 1: return x >= 0x1p-32f && x <= 0x1p32f;  // [2^-32, 2^32]
    case 2:
    case 3: return ax < 16.f;                       // |x| < 16 (the engines: angles)
    case 4: return x >= 0x1p-57f && x <= 0x1p40f;   // [2^-57, 2^40]: below 2^-57 the result is subnormal (not tabulated)
    case 5: return x > 0.f && x <= 0x1p40f;         // (0, 2^40], subnormal inputs included
    default: return x >= 0.f && x <= 1.f;           // 6 / 7: the exponent in [0, 1]
    }
}
const float kPowBaseThr = 0.03f / 0.30f, kPowBaseHalo = 0.03f / 0.6f;  // (sk_core.cpp / r6: the same expressions)

}  // namespace pmm