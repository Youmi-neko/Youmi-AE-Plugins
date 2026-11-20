// Tebure - motion model (see tb_model.h, docs\shake_model.md).
#include "tb_model.h"
#include "platform/plat_math.h"  // tbm:: = the libm calls that feed the picture (forward to std::)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <string>

namespace tb {

static std::map<std::string, std::array<double, 6>>& PeakCache()
{
    static std::map<std::string, std::array<double, 6>> c;
    return c;
}
static std::mutex& PeakCacheMutex()
{
    static std::mutex m;
    return m;
}

static const double kPi = 3.14159265358979323846;
static const double kTwoPi = 2.0 * kPi;

uint64_t Mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
double Uni(uint64_t seed, uint64_t salt, int64_t i)
{
    const uint64_t h = Mix64(Mix64(seed * 0x100000001B3ull + salt) ^ (uint64_t)i * 0xD6E8FEB86659FD93ull);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}
double Gauss(uint64_t seed, uint64_t salt, int64_t i)
{
    const double u1 = std::max(1e-12, Uni(seed, salt, 2 * i)), u2 = Uni(seed, salt, 2 * i + 1);
    return std::sqrt(-2.0 * tbm::log(u1)) * tbm::cos(kTwoPi * u2);
}
static inline int64_t FloorI(double v) { return (int64_t)std::floor(v); }
static inline int64_t Mod(int64_t a, int64_t m) { return m > 0 ? ((a % m) + m) % m : a; }

Pose Lerp(const Pose& a, const Pose& b, double u)
{
    Pose r;
    for (int i = 0; i < 6; ++i)
        r.set(i, a.get(i) + (b.get(i) - a.get(i)) * u);
    return r;
}

double SoftLimit(double x, double L)
{
    if (!(L > 0))
        return 0.0;
    const double u = x / L, u2 = u * u, u4 = u2 * u2, u8 = u4 * u4;
    return x / tbm::pow(1.0 + u8, 0.125);
}

double Model::Sines::eval(double t) const
{
    double v = 0;
    const size_t n = f.size();
    for (size_t k = 0; k < n; ++k)
        v += a[k] * tbm::sin(kTwoPi * f[k] * t + ph[k]);
    return v;
}

namespace {
typedef double (*Spec)(double f, const double* p);
// unit-RMS sum of M sinusoids, log-spaced frequencies (hashed jitter), amplitudes from the PSD S(f)
Model::Sines MakeSines(uint64_t seed, uint64_t salt, double flo, double fhi, int M, Spec S, const double* sp, double tiltB, double loop)
{
    Model::Sines s;
    if (!(fhi > flo) || !(flo > 0) || M <= 0)
        return s;
    const double L0 = tbm::log(flo), L1 = tbm::log(fhi), d = (L1 - L0) / M;
    double sum2 = 0;
    for (int k = 0; k < M; ++k)
    {
        const double lf = L0 + (k + Uni(seed, salt, 2 * k)) * d;
        const double f = tbm::exp(lf);
        const double w = S(f, sp) * f * d;
        double a = std::sqrt(std::max(0.0, w));
        a *= tbm::pow(f / 1.5, 0.8 * tiltB);  // balance: spectral tilt, RMS renormalised below
        if (!(a > 0) || !std::isfinite(a))
            continue;
        double fq = f;
        if (loop > 0)
            fq = std::max(1.0, std::floor(f * loop + 0.5)) / loop;
        s.f.push_back(fq);
        s.a.push_back(a);
        s.ph.push_back(kTwoPi * Uni(seed, salt, 2 * k + 1));
        sum2 += a * a;
    }
    if (!(sum2 > 0))
        return Model::Sines();
    const double g = std::sqrt(2.0 / sum2);
    for (double& a : s.a)
    {
        a *= g;
        s.rmsBound += std::fabs(a);
    }
    return s;
}
// p = {f0, Q, rho, fhp}
double SpecHand(double f, const double* p)
{
    const double u = f / p[0], u2 = u * u;
    const double res = 1.0 / ((1 - u2) * (1 - u2) + u2 / (p[1] * p[1]));
    const double v = f / p[3], hp = v * v / (1 + v * v);
    return res * tbm::pow(1 + u2, p[2]) * hp;
}
// p = {f0, sigma} gaussian in Hz
double SpecGaussHz(double f, const double* p)
{
    const double z = (f - p[0]) / p[1];
    return tbm::exp(-0.5 * z * z);
}
// p = {f0, b} gaussian in log f, equal energy per log bin
double SpecLogGauss(double f, const double* p)
{
    const double z = tbm::log(f / p[0]) / p[1];
    return tbm::exp(-0.5 * z * z) / f;
}
// p = {fc}  1/f^2 above fc (random-walk like drift), flat below
double SpecDrift(double f, const double* p) { return 1.0 / (f * f + p[0] * p[0]); }
double SpecFlatLog(double f, const double*) { return 1.0 / f; }

inline double Gain(double fchar, double b) { return std::min(4.0, std::max(0.25, tbm::pow(std::max(1e-3, fchar) / 1.5, 0.8 * b))); }

// minimum-jerk 0..1 with a small overshoot
inline double MinJerk(double u)
{
    if (u <= 0)
        return 0;
    if (u >= 1)
        return 1;
    const double m = u * u * u * (10 - 15 * u + 6 * u * u);
    return m + 0.08 * tbm::sin(kPi * u) * tbm::sin(kPi * u) * u;
}
// under-damped second-order step response, peak at u = 1, overshoot o
inline double Settle(double u, double o)
{
    if (u <= 0)
        return 0;
    if (u > 12)
        return 1;
    o = std::min(0.6, std::max(0.0005, o));
    const double lo = tbm::log(o), z = -lo / std::sqrt(kPi * kPi + lo * lo);
    const double wd = kPi, wn = wd / std::sqrt(1 - z * z);
    return 1 - tbm::exp(-z * wn * u) * (tbm::cos(wd * u) + z * wn / wd * tbm::sin(wd * u));
}
}  // namespace

void Model::Build(const Shape& s, const Controls& c, double loopTau, int64_t loopGrid, double loopPhase)
{
    sh_ = s;
    c_ = c;
    loopTau_ = loopTau > 0 ? loopTau : 0;
    loopGrid_ = loopGrid > 0 ? loopGrid : 0;
    loopPhase_ = loopPhase > 0 ? loopPhase : 0;
    seed_ = Mix64(0x7E8B5E00ull + c.seed);
    const double L = loopTau_;
    const double bx = c.balPos, br = c.balRot, bz = c.balZoom;
    handX_ = handY_ = handR_ = bodyX_ = bodyY_ = tremX_ = tremY_ = breathPh_ = driftX_ = driftY_ = Sines();
    vibX_ = vibY_ = vibR_ = bigX_ = bigY_ = bigR_ = bigZ_ = swell_ = weaveX_ = weaveY_ = gaitJit_ = gaitAmp_ = grit_ = Sines();
    warpX_ = warpY_ = amodX_ = amodY_ = Sines();
    const uint64_t sd = seed_;
    if (s.handDeg > 0 || s.bodyPx > 0)
    {
        const double f0 = std::max(0.01, s.handHz);
        const double rho = std::min(1.0, std::max(-1.0, s.handRough));
        double p[4] = {f0, std::max(0.3, s.handQ), rho, f0 / 4};
        const double lo = f0 / 16, hi = std::min(f0 * 30, 45.0);
        if (s.handDeg > 0)
        {
            const double fy = f0 * std::min(8.0, std::max(0.125, s.handYHz));
            double py[4] = {fy, p[1], rho, fy / 4};
            handX_ = MakeSines(sd, 11, lo, hi, 96, SpecHand, p, bx, L);
            handY_ = MakeSines(sd, 12, fy / 16, std::min(fy * 30, 45.0), 96, SpecHand, py, bx, L);
            handR_ = MakeSines(sd, 13, lo, hi, 96, SpecHand, p, br, L);
            if (s.handIrr > 0)
            {
                // irregularity: a slow time warp per axis (arcs of varying length, the timing drifts) and a
                // per-arc amplitude modulation whose dips read as short pauses
                double w[4] = {f0 * 0.2, 0.6, 0.0, f0 * 0.2 / 8};
                double a[4] = {f0 * 0.45, 0.6, 0.0, f0 * 0.45 / 8};
                warpX_ = MakeSines(sd, 16, f0 * 0.02, f0 * 3, 32, SpecHand, w, 0, L);
                warpY_ = MakeSines(sd, 17, f0 * 0.02, f0 * 3, 32, SpecHand, w, 0, L);
                amodX_ = MakeSines(sd, 18, f0 * 0.02, f0 * 3, 32, SpecHand, a, 0, L);
                amodY_ = MakeSines(sd, 19, f0 * 0.02, f0 * 3, 32, SpecHand, a, 0, L);
            }
        }
        if (s.bodyPx > 0)
        {
            const double fb = std::max(0.01, s.bodyHz);
            double q[4] = {fb, 0.6, std::min(0.0, rho), fb / 8};
            bodyX_ = MakeSines(sd, 14, fb / 16, std::min(fb * 30, 45.0), 64, SpecHand, q, bx, L);
            bodyY_ = MakeSines(sd, 15, fb / 16, std::min(fb * 30, 45.0), 64, SpecHand, q, bx, L);
        }
    }
    if (s.tremorDeg > 0)
    {
        double p[2] = {9.5, 1.3};
        tremX_ = MakeSines(sd, 21, 6, 14, 32, SpecGaussHz, p, bx, L);
        tremY_ = MakeSines(sd, 22, 6, 14, 32, SpecGaussHz, p, bx, L);
    }
    breathHzQ_ = std::max(0.01, s.breathHz);
    if (L > 0)
        breathHzQ_ = std::max(1.0, std::floor(breathHzQ_ * L + 0.5)) / L;
    if (s.breathDeg > 0)
    {
        double p[1] = {0.05};
        breathPh_ = MakeSines(sd, 31, 0.01, 0.2, 12, SpecDrift, p, 0, L);
    }
    if (s.driftDeg > 0)
    {
        double p[1] = {0.03};
        driftX_ = MakeSines(sd, 41, 0.01, 1.5, 48, SpecDrift, p, bx, L);
        driftY_ = MakeSines(sd, 42, 0.01, 1.5, 48, SpecDrift, p, bx, L);
    }
    if (s.vibPx > 0)
    {
        double p[2] = {std::max(0.1, s.vibHz), std::max(0.01, s.vibBand)};
        const double lo = p[0] * tbm::exp(-3 * p[1]), hi = p[0] * tbm::exp(3 * p[1]);
        vibX_ = MakeSines(sd, 51, lo, hi, 48, SpecLogGauss, p, bx, L);
        vibY_ = MakeSines(sd, 52, lo, hi, 48, SpecLogGauss, p, bx, L);
        vibR_ = MakeSines(sd, 53, lo, hi, 48, SpecLogGauss, p, br, L);
    }
    if (s.bigPx > 0 || s.bigRot > 0 || s.bigZoom > 0)
    {
        const double f0 = std::max(0.005, s.bigHz);
        double p[4] = {f0, 0.6, 0.0, f0 / 10};
        bigX_ = MakeSines(sd, 61, f0 / 20, f0 * 8, 64, SpecHand, p, bx, L);
        bigY_ = MakeSines(sd, 62, f0 / 20, f0 * 8, 64, SpecHand, p, bx, L);
        bigR_ = MakeSines(sd, 63, f0 / 20, f0 * 8, 64, SpecHand, p, br, L);
        bigZ_ = MakeSines(sd, 64, f0 / 20, f0 * 8, 64, SpecHand, p, bz, L);
    }
    if (s.swell > 0)
    {
        double p[4] = {0.07, 0.5, 0.0, 0.01};
        swell_ = MakeSines(sd, 71, 0.005, 0.5, 16, SpecHand, p, 0, L);
    }
    if (s.weavePx > 0)
    {
        double p[4] = {0.8, 0.7, 0.3, 0.1};
        weaveX_ = MakeSines(sd, 81, 0.05, 6, 48, SpecHand, p, bx, L);
        weaveY_ = MakeSines(sd, 82, 0.05, 6, 48, SpecHand, p, bx, L);
    }
    if (s.bouncePx > 0 || s.swayPx > 0 || s.walkRoll > 0 || s.walkPan > 0 || s.heelPx > 0)
    {
        double p[1] = {0.1};
        gaitJit_ = MakeSines(sd, 91, 0.02, 1.0, 12, SpecDrift, p, 0, L);
        double q[4] = {0.4, 0.6, 0.0, 0.02};
        gaitAmp_ = MakeSines(sd, 92, 0.02, 2.0, 12, SpecHand, q, 0, loopPhase_);  // in steps (phase domain)
    }
    if (s.impGrit > 0 && s.impPx > 0)
        grit_ = MakeSines(sd, 101, 4, 20, 24, SpecFlatLog, nullptr, bx, 0);
    gainPos_ = 1;
    gainRot_ = 1;
    gainZoom_ = 1;
    // band limit: gain = T(f * S) / T(f) (<= 1), T = 1 below 0.4 fN fading (cos^2) to 0 at 0.75 fN; components already
    // above Nyquist at 100 % (T(f) ~ 0: e.g. engine vibration, deliberately aliased like a short shutter) are kept
    bandS_ = (c.bandSpeed > 1.0 + 1e-9 && c.nyquist > 0) ? c.bandSpeed : 0;
    bandN_ = c.nyquist;
    if (bandS_ > 0)
        for (Sines* sp : {&handX_, &handY_, &handR_, &bodyX_, &bodyY_, &tremX_, &tremY_, &driftX_, &driftY_, &vibX_, &vibY_, &vibR_,
                          &bigX_, &bigY_, &bigR_, &bigZ_, &weaveX_, &weaveY_})
            for (size_t k = 0; k < sp->f.size(); ++k)
                sp->a[k] *= BandGain(sp->f[k]);

    // ---- bounds (per unit amount / multipliers): stationary RMS and deterministic peaks per axis
    for (int i = 0; i < 6; ++i)
        rmsPos_[i] = peak_[i] = trigPeak_[i] = 0;
    const double kpx = 1.0;
    auto addR = [&](int i, double v) { rmsPos_[i] += v * v; };
    const double fam = std::sqrt(1 + 0.42 * std::min(1.0, std::max(0.0, s.handIrr)));  // per-arc amplitude modulation
    addR(0, s.handDeg * fam);
    addR(1, s.handDeg * s.handYX * fam);
    addR(2, s.handDeg * s.handRoll * fam);
    addR(3, s.bodyPx * kpx);
    addR(4, s.bodyPx * kpx);
    addR(0, s.tremorDeg);
    addR(1, s.tremorDeg);
    addR(3, s.vibPx);
    addR(4, s.vibPx * 1.3);
    addR(0, s.vibPx * 0.004);
    addR(1, s.vibPx * 0.004);
    addR(2, s.vibPx * 0.01);
    addR(3, s.bigPx);
    addR(4, s.bigPx * 0.8);
    addR(2, s.bigRot);
    addR(5, s.bigZoom);
    addR(3, s.weavePx);
    addR(4, s.weavePx * 0.6);
    // the swell multiplies a unit-RMS process by (1 + swell * N): RMS factor sqrt(1 + swell^2)
    const double sw = 1.15 * std::sqrt(1 + s.swell * s.swell);
    for (int i = 0; i < 6; ++i)
        rmsPos_[i] = std::sqrt(rmsPos_[i]) * sw;
    // deterministic / event peaks
    peak_[1] += 1.3 * s.breathDeg;
    peak_[0] += 3.5 * s.driftDeg;
    peak_[1] += 3.5 * s.driftDeg * s.handYX;
    peak_[4] += 1.15 * s.bouncePx + 1.6 * s.heelPx + 1.5 * s.pulsePct;
    peak_[3] += 1.15 * s.swayPx + 0.3 * s.heelPx;
    peak_[2] += 1.15 * s.walkRoll + 0.03 * s.heelPx;
    peak_[0] += 1.15 * s.walkPan;
    peak_[5] += 2.0 * s.pulsePct;
    peak_[4] += 2.2 * s.bumpPx;
    peak_[2] += 0.02 * s.bumpPx;
    {
        const double jx = s.joltMode == JOLT_V ? 0 : s.joltPx * 0.7071 * 3.2 * 1.6;
        const double jy = s.joltMode == JOLT_H ? 0 : s.joltPx * 0.7071 * 3.2 * 1.6 * s.joltYX;
        peak_[3] += jx;
        peak_[4] += jy;
        peak_[2] += 0.003 * jx;
    }
    {
        const double yx = std::max(0.0, s.jitterYX), ax = s.jitterPx * 2 / (1 + yx), ay = ax * yx;
        peak_[3] += 6.0 * ax;
        peak_[4] += 6.0 * ay;
    }
    // one trigger
    if (s.impPx > 0 || s.impRot > 0 || s.impZoom > 0)
    {
        double m = 0;
        if (s.impType == IMP_PHYSICAL)
            m = (0.35 + 1 + 0.55 + 0.3) / 1.25 + 3.0 * s.impGrit;
        else if (s.impType == IMP_GAMENDO)
            m = 1 + 0.5 * 3.0 * s.impRandom;
        else
            m = 1.2;
        trigPeak_[3] = trigPeak_[4] = s.impPx * m;
        trigPeak_[2] = s.impRot * m;
        trigPeak_[5] = s.impZoom * (s.impType == IMP_PHYSICAL ? 1.2 : m);
        if (s.impType != IMP_GAMENDO)
        {
            // measured peak of one response over 16 trigger identities (directions), +15 % margin. A pure function
            // of the impact fields + balances: memoised process-wide (thread safe, never changes a result)
            const double keyv[13] = {(double)s.impType, s.impPx, s.impRot, s.impZoom, s.impHz, s.impGrit, s.impAngle, s.impRandom,
                                     s.decay, c.balPos, c.balRot, c.balZoom, (double)c.seed};
            const std::string key((const char*)keyv, sizeof(keyv));
            {
                std::lock_guard<std::mutex> lk(PeakCacheMutex());
                auto it = PeakCache().find(key);
                if (it != PeakCache().end())
                {
                    for (int i = 0; i < 6; ++i)
                        trigPeak_[i] = it->second[i];
                    return;
                }
            }
            double mx[6] = {0, 0, 0, 0, 0, 0};
            std::vector<Trigger> tv(1);
            Ctx c;
            const double len = std::min(ResponseLength(), 8.0);
            for (int id = 0; id < 16; ++id)
            {
                tv[0].id = id * 7919 + 1;
                tv[0].tau = 0;
                tv[0].strength = 1;
                for (double d = 0; d <= len; d += 0.002)
                {
                    c.tau = d;
                    Pose p;
                    Impact(c, tv, p);
                    for (int i = 0; i < 6; ++i)
                        mx[i] = std::max(mx[i], std::fabs(p.get(i)));
                }
            }
            std::array<double, 6> v;
            for (int i = 0; i < 6; ++i)
                v[i] = trigPeak_[i] = 1.15 * mx[i];
            std::lock_guard<std::mutex> lk(PeakCacheMutex());
            if (PeakCache().size() > 4096)
                PeakCache().clear();
            PeakCache()[key] = v;
        }
    }
}

double Model::Taper(double fr) const
{
    const double x = fr / bandN_;
    if (x <= 0.4)
        return 1;
    if (x >= 0.75)
        return 0;
    const double c = tbm::cos(0.5 * 3.14159265358979 * (x - 0.4) / 0.35);
    return c * c;
}
double Model::BandGain(double f) const
{
    if (!(bandS_ > 0))
        return 1;
    const double t1 = Taper(f);
    if (t1 < 0.02)
        return 1;
    return std::min(1.0, Taper(f * bandS_) / t1);
}

double Model::ResponseLength() const
{
    const Shape& s = sh_;
    if (s.quakeEnv > 0)
        return s.decay > 0 ? 1.8 + 8 * s.decay : 1e9;
    if (!(s.impPx > 0 || s.impRot > 0 || s.impZoom > 0))
        return 0;
    if (s.impType == IMP_WHIP)
        return 3.5 * (s.decay > 0 ? s.decay : 1.0);
    if (!(s.decay > 0))
        return 1e9;
    return s.impType == IMP_PHYSICAL ? 7 * 1.6 * s.decay : 7 * s.decay;
}

Pose Model::Bound(double amountMax, const double mulMax[4], double trigMax, bool multiHit, bool forAutoScale) const
{
    Pose b;
    const int axMul[6] = {0, 1, 2, 0, 1, 3};
    for (int i = 0; i < 6; ++i)
    {
        // whips are events that briefly leave the frame: the auto scale ignores them (the edge is repeated then)
        const double tp = (forAutoScale && sh_.impType == IMP_WHIP) ? 0.0 : trigPeak_[i];
        const double qk = sh_.quakeEnv > 0 && (i == 3 || i == 4 || i == 2 || i == 5) ? 1.2 * std::max(1.0, trigMax) : 1.0;
        const double v = 3.0 * rmsPos_[i] * qk + peak_[i] + (multiHit ? 1.5 : 1.0) * tp * std::max(0.0, trigMax);
        b.set(i, 1.1 * v * std::max(0.0, amountMax) * std::max(0.0, mulMax[axMul[i]]));
    }
    return b;
}

double Model::Swell(double tau) const
{
    if (swell_.empty())
        return 1;
    return std::max(0.25, 1 + sh_.swell * swell_.eval(tau));
}

// cell lattice: one candidate event per cell of length `cell` (periodic with the loop)
double Model::CellTime(int64_t k, double cell, uint64_t salt, bool& occupied) const
{
    int64_t hk = k;
    double c = cell;
    if (loopTau_ > 0)
    {
        const int64_t K = std::max<int64_t>(1, (int64_t)std::floor(loopTau_ / cell + 0.5));
        c = loopTau_ / K;
        hk = Mod(k, K);
    }
    occupied = Uni(seed_, salt, hk * 3) < 0.8;
    return (k + 0.1 + 0.8 * Uni(seed_, salt, hk * 3 + 1)) * c;
}

void Model::Corrections(double tau, double& x, double& y) const
{
    const Shape& s = sh_;
    x = y = 0;
    if (driftX_.empty())
        return;
    double cell = std::max(0.2, s.corrInt);
    if (loopTau_ > 0)
        cell = loopTau_ / std::max<int64_t>(1, (int64_t)std::floor(loopTau_ / cell + 0.5));
    // last two occupied events at or before tau
    int64_t ev[2] = {INT64_MIN, INT64_MIN};
    double et[2] = {0, 0};
    int found = 0;
    for (int64_t k = FloorI(tau / cell); k > FloorI(tau / cell) - 24 && found < 2; --k)
    {
        bool occ = false;
        const double t = CellTime(k, std::max(0.2, s.corrInt), 201, occ);
        if (occ && t <= tau)
        {
            ev[found] = k;
            et[found] = t;
            ++found;
        }
    }
    const double dx = s.driftDeg, dy = s.driftDeg * s.handYX;
    double Dx = driftX_.eval(tau) * dx, Dy = driftY_.eval(tau) * dy;
    if (found == 0)
    {
        x = Dx;
        y = Dy;
        return;
    }
    auto target = [&](int i, double& cx, double& cy) {
        if (i >= found)
        {
            cx = cy = 0;
            return;
        }
        const int64_t hk = loopTau_ > 0 ? Mod(ev[i], (int64_t)std::floor(loopTau_ / cell + 0.5)) : ev[i];
        cx = driftX_.eval(et[i]) * dx - 0.15 * dx * Gauss(seed_, 202, hk);
        cy = driftY_.eval(et[i]) * dy - 0.15 * dy * Gauss(seed_, 203, hk);
    };
    double c0x, c0y, c1x, c1y;
    target(0, c0x, c0y);
    target(1, c1x, c1y);
    const int64_t hk0 = loopTau_ > 0 ? Mod(ev[0], (int64_t)std::floor(loopTau_ / cell + 0.5)) : ev[0];
    const double d = std::min(0.2 * cell, 0.28 + 0.15 * Uni(seed_, 204, hk0));
    const double S = MinJerk((tau - et[0]) / d);
    x = Dx - (c1x + (c0x - c1x) * S);
    y = Dy - (c1y + (c0y - c1y) * S);
}

void Model::Gait(const Ctx& x, Pose& p) const
{
    const Shape& s = sh_;
    if (gaitJit_.empty())
        return;
    const int64_t kp = loopPhase_ > 0 ? (int64_t)std::floor(loopPhase_ + 0.5) : 0;
    auto hk = [&](int64_t k) { return kp > 0 ? Mod(k, kp) : k; };
    // per-step variability (gaitVar): each step's timing offset and size are random, blended smoothly across
    // the step; left / right asymmetry; ~3 % of steps stumble (late, harder landing)
    const double gv = std::min(1.0, std::max(0.0, s.gaitVar));
    const double asym = gv * 0.12 * Gauss(seed_, 310, 0);
    auto stepOff = [&](int64_t k) {
        double o = gv * 0.13 * Gauss(seed_, 311, hk(k));
        if (Uni(seed_, 312, hk(k)) < 0.03 * gv)
            o += 0.12;
        return o;
    };
    auto stepAmp = [&](int64_t k) { return (1 + gv * 0.2 * Gauss(seed_, 313, hk(k))) * (1 + asym * (Mod(k, 2) == 0 ? 1 : -1)); };
    double Ph = x.phase + (0.04 + 0.08 * std::min(1.0, std::max(0.0, s.gaitVar))) * gaitJit_.eval(x.tau);
    double av = 1 + 0.12 * gaitAmp_.eval(Ph);
    if (gv > 0)
    {
        const int64_t k0 = FloorI(Ph);
        const double u = Ph - (double)k0, sm = u * u * (3 - 2 * u);
        Ph += stepOff(k0) + (stepOff(k0 + 1) - stepOff(k0)) * sm;
        av *= stepAmp(k0) + (stepAmp(k0 + 1) - stepAmp(k0)) * sm;
    }
    const double r = std::min(1.0, std::max(0.0, s.heelSharp));
    const double h = (1 - r) * (-tbm::cos(kTwoPi * Ph)) + r * (2 * std::fabs(tbm::sin(kPi * Ph)) - 1);
    // band limit of the rhythm (0.3): a cadence pushed above Nyquist by the Period fades instead of strobing
    const double bg = BandGain(std::max(0.05, x.rate)), bgs = BandGain(std::max(0.05, x.rate * 0.5));
    const double gb = Gain(std::max(0.1, x.rate), c_.balPos) * bg, gs = Gain(std::max(0.05, x.rate * 0.5), c_.balPos) * bgs;
    p.ty += s.bouncePx * av * h * gb;
    p.tx += s.swayPx * av * tbm::sin(kPi * Ph + 0.6) * gs;
    p.roll += s.walkRoll * av * tbm::sin(kPi * Ph + 0.25) * Gain(std::max(0.05, x.rate * 0.5), c_.balRot);
    p.pan += s.walkPan * av * tbm::sin(kPi * Ph + 1.1) * gs;
    if (s.heelPx > 0)
    {
        const double fh = 9 - 3 * r, tauh = 0.07 + 0.05 * r;
        const double gh = Gain(fh, c_.balPos) * BandGain(fh);
        for (int64_t k = FloorI(Ph); k >= FloorI(Ph) - 1; --k)
        {
            const double d = (Ph - (double)k) / std::max(0.05, x.rate);
            if (d < 0 || d > 0.8)
                continue;
            const bool stumble = gv > 0 && Uni(seed_, 312, hk(k)) < 0.03 * gv;
            const double amp = s.heelPx * (1 + 0.3 * Gauss(seed_, 301, hk(k))) * gh * (stumble ? 2.5 : 1.0) * (gv > 0 ? stepAmp(k) : 1.0);
            const double resp = tbm::exp(-d / tauh) * tbm::sin(kTwoPi * fh * d);
            const double side = (Mod(k, 2) == 0) ? 1.0 : -1.0;
            p.ty += amp * resp;
            p.tx += side * amp * 0.15 * resp;
            p.roll += side * amp * 0.012 * resp;
        }
    }
    (void)x;
}

void Model::Bumps(double tau, Pose& p) const
{
    const Shape& s = sh_;
    if (!(s.bumpPx > 0))
        return;
    const double cell = 1.0 / std::max(0.01, s.bumpRate);
    double c = cell;
    if (loopTau_ > 0)
        c = loopTau_ / std::max<int64_t>(1, (int64_t)std::floor(loopTau_ / cell + 0.5));
    const double win = 3.5;
    const double g = Gain(1.6, c_.balPos);
    for (int64_t k = FloorI(tau / c); k >= FloorI((tau - win) / c) - 1; --k)
    {
        bool occ = false;
        const double t = CellTime(k, cell, 401, occ);
        const double d = tau - t;
        if (!occ || d < 0 || d > win)
            continue;
        const int64_t hk = loopTau_ > 0 ? Mod(k, (int64_t)std::floor(loopTau_ / cell + 0.5)) : k;
        const double amp = s.bumpPx * (0.5 + Uni(seed_, 402, hk)) * g;
        const double resp = tbm::exp(-d / 0.55) * tbm::sin(kTwoPi * 1.6 * d) + 0.5 * tbm::exp(-d / 0.05) * tbm::sin(kTwoPi * 12 * d);
        const double sg = Uni(seed_, 403, hk) < 0.5 ? -1 : 1;
        p.ty += amp * resp;
        p.roll += sg * amp * 0.008 * resp;
        p.tx += sg * amp * 0.1 * resp;
    }
}

void Model::Jolts(double tau, Pose& p) const
{
    const Shape& s = sh_;
    if (!(s.joltPx > 0))
        return;
    const double cell = std::max(0.05, s.joltInt);
    double c = cell;
    int64_t K = 0;
    if (loopTau_ > 0)
    {
        K = std::max<int64_t>(1, (int64_t)std::floor(loopTau_ / cell + 0.5));
        c = loopTau_ / K;
    }
    const double sg = s.joltPx * 0.70710678;
    const double yx = std::max(0.0, s.joltYX);
    const double py = s.joltMode == JOLT_RECT ? yx / (1 + yx) : 0;
    auto hidx = [&](int64_t k) { return K > 0 ? Mod(k, K) : k; };
    auto axisOf = [&](int64_t k) { return Uni(seed_, 503, hidx(k)) < py ? 1 : 0; };
    // target of event k
    auto target = [&](int64_t k, double& X, double& Y) {
        const int64_t hk = hidx(k);
        X = Y = 0;
        if (s.joltMode == JOLT_RECT)
        {
            // each event moves one axis; the other keeps the value of the last event that moved it
            bool gotX = false, gotY = false;
            for (int64_t j = k; j > k - 48 && !(gotX && gotY); --j)
            {
                bool occ = false;
                CellTime(j, cell, 501, occ);
                if (!occ)
                    continue;
                const int a = axisOf(j);
                if (a == 0 && !gotX)
                {
                    X = sg * Gauss(seed_, 504, hidx(j));
                    gotX = true;
                }
                if (a == 1 && !gotY)
                {
                    Y = sg * Gauss(seed_, 505, hidx(j));
                    gotY = true;
                }
            }
            return;
        }
        if (s.joltMode != JOLT_V)
            X = sg * Gauss(seed_, 504, hk);
        if (s.joltMode != JOLT_H)
            Y = sg * yx * Gauss(seed_, 505, hk);
    };
    int64_t ev[4];
    double et[4];
    int found = 0;
    for (int64_t k = FloorI(tau / c); k > FloorI(tau / c) - 64 && found < 4; --k)
    {
        bool occ = false;
        const double t = CellTime(k, cell, 501, occ);
        if (occ && t <= tau)
        {
            ev[found] = k;
            et[found] = t;
            ++found;
        }
    }
    if (found == 0)
        return;
    // position = oldest target + sum of the later steps, each eased by its own settle curve (telescopes to the
    // newest target once every transition has settled)
    double TX[4], TY[4];
    for (int i = 0; i < found; ++i)
        target(ev[i], TX[i], TY[i]);
    double X = TX[found - 1], Y = TY[found - 1];
    for (int i = found - 2; i >= 0; --i)
    {
        const double d = std::max(0.01, s.joltDur) * (0.7 + 0.6 * Uni(seed_, 506, hidx(ev[i])));
        const double S = Settle((tau - et[i]) / d, s.joltSettle);
        X += (TX[i] - TX[i + 1]) * S;
        Y += (TY[i] - TY[i + 1]) * S;
    }
    const double g = Gain(1.0 / (2 * std::max(0.01, s.joltDur)), c_.balPos);
    p.tx += X * g;
    p.ty += Y * g;
    p.roll += 0.003 * X * Gain(2.0, c_.balRot);
}

void Model::Film(const Ctx& x, Pose& p) const
{
    const Shape& s = sh_;
    if (!weaveX_.empty())
    {
        p.tx += s.weavePx * weaveX_.eval(x.tau);
        p.ty += s.weavePx * 0.6 * weaveY_.eval(x.tau);
    }
    if (s.jitterPx > 0)
    {
        int64_t n = x.gridIsComp ? FloorI(x.tauFilm * 24 + 1e-6) : x.grid;
        if (loopGrid_ > 0)
            n = Mod(n, loopGrid_);
        const double yx = std::max(0.0, s.jitterYX), ax = s.jitterPx * 2 / (1 + yx), ay = ax * yx;
        const double hop = Uni(seed_, 603, n) < 0.04 ? 2.5 : 1.0;
        p.tx += ax * Gauss(seed_, 601, n) * hop;
        p.ty += ay * Gauss(seed_, 602, n) * hop;
    }
}

void Model::Impact(const Ctx& x, const std::vector<Trigger>& trig, Pose& p) const
{
    const Shape& s = sh_;
    if (!(s.impPx > 0 || s.impRot > 0 || s.impZoom > 0) || trig.empty())
        return;
    const double len = ResponseLength();
    const double D = s.decay;
    const double ang0 = s.impAngle * kPi / 180;
    for (const Trigger& tg : trig)
    {
        const double d = x.tau - tg.tau;
        if (d < 0 || d > len)
            continue;
        const double g = tg.strength;
        const uint64_t hs = Mix64(seed_ ^ (uint64_t)(tg.id * 0x9E37ull + 77));
        if (s.impType == IMP_PHYSICAL)
        {
            static const double fm[4] = {0.45, 1, 1.7, 2.9}, am[4] = {0.35, 1, 0.55, 0.3}, tm[4] = {1.6, 1, 0.6, 0.35};
            double vx = 0, vy = 0, vr = 0;
            for (int m = 0; m < 4; ++m)
            {
                const double f = s.impHz * fm[m];
                const double env = D > 0 ? tbm::exp(-d / (D * tm[m])) : 1.0;
                const double w = am[m] * env * tbm::sin(kTwoPi * f * d) * Gain(f, c_.balPos) / 1.25;
                const double a = ang0 + s.impRandom * kPi * (Uni(hs, 701, m) - 0.5);
                const double sg = Uni(hs, 702, m) < 0.5 ? -1.0 : 1.0;
                vx += sg * w * tbm::cos(a);
                vy += sg * w * tbm::sin(a);
                vr += (Uni(hs, 703, m) < 0.5 ? -1.0 : 1.0) * am[m] * env * tbm::sin(kTwoPi * f * d + 0.4) / 1.25;
            }
            if (!grit_.empty())
            {
                const double env = D > 0 ? tbm::exp(-d / (0.6 * D)) : 1.0;
                const double a = Uni(hs, 704, 0) * kTwoPi;
                const double gv = s.impGrit * env * grit_.eval(d + 100 * Uni(hs, 705, 0));
                vx += gv * tbm::cos(a);
                vy += gv * tbm::sin(a);
            }
            p.tx += s.impPx * g * vx;
            p.ty += s.impPx * g * vy;
            p.roll += s.impRot * g * vr * Gain(s.impHz, c_.balRot);
            const double ez = D > 0 ? tbm::exp(-d / (0.8 * D)) : 1.0;
            p.zoom += s.impZoom * g * ez * tbm::sin(kTwoPi * 0.6 * s.impHz * d + 0.3) * Gain(s.impHz * 0.6, c_.balZoom);
        }
        else if (s.impType == IMP_GAMENDO)
        {
            const int64_t m = x.grid - tg.grid;
            if (m < 0)
                continue;
            const double env = D > 0 ? tbm::exp(-d / D) : 1.0;
            const double sgn = (m % 2 == 0) ? 1.0 : -1.0;
            const double amp = sgn * env * (1 + 0.5 * s.impRandom * Gauss(hs, 711, m));
            const double a = ang0 + s.impRandom * (kPi / 3) * Gauss(hs, 712, m) * 0.5;
            p.tx += s.impPx * g * amp * tbm::cos(a);
            p.ty += s.impPx * g * amp * tbm::sin(a);
            p.roll += s.impRot * g * amp;
            p.zoom += s.impZoom * g * std::fabs(amp);
        }
        else
        {
            const double Dw = D > 0 ? D : 1.0;
            const double u = d / Dw;
            double sh;
            if (u < 1)
                sh = tbm::sin(kPi * u) * tbm::sin(kPi * u);
            else
                sh = -0.18 * tbm::exp(-3 * (u - 1)) * tbm::sin(kTwoPi * 1.2 * (u - 1));
            const double a = ang0 + s.impRandom * kPi * (Uni(hs, 721, 0) - 0.5);
            const double sg = Uni(hs, 722, 0) < 0.5 ? -1.0 : 1.0;
            p.tx += s.impPx * g * sg * sh * tbm::cos(a);
            p.ty += s.impPx * g * sg * sh * tbm::sin(a);
            p.roll += s.impRot * g * sg * sh;
            p.zoom += s.impZoom * g * sh;
        }
    }
}

Pose Model::Raw(const Ctx& x, const std::vector<Trigger>& trig) const
{
    const Shape& s = sh_;
    Pose p;
    const double m = Swell(x.tau);
    if (!handX_.empty())
    {
        double tx = x.tau, ty = x.tau, ax = 1, ay = 1;
        if (!warpX_.empty())
        {
            const double irr = std::min(1.0, std::max(0.0, s.handIrr));
            const double k = irr * 0.3 / std::max(0.01, s.handHz);
            tx += k * warpX_.eval(x.tau);
            ty += k * warpY_.eval(x.tau);
            ax = std::max(0.08, 1 + irr * 0.65 * amodX_.eval(x.tau));
            ay = std::max(0.08, 1 + irr * 0.65 * amodY_.eval(x.tau));
        }
        const double hx = handX_.eval(tx);
        p.pan += s.handDeg * m * ax * hx;
        p.tilt += s.handDeg * s.handYX * m * ay * handY_.eval(ty);
        p.roll += s.handDeg * s.handRoll * m * ax * (0.8 * handR_.eval(tx) + 0.6 * hx);
    }
    if (!bodyX_.empty())
    {
        p.tx += s.bodyPx * m * bodyX_.eval(x.tau);
        p.ty += s.bodyPx * m * bodyY_.eval(x.tau);
    }
    if (!tremX_.empty())
    {
        p.pan += s.tremorDeg * tremX_.eval(x.tau);
        p.tilt += s.tremorDeg * tremY_.eval(x.tau);
    }
    if (s.breathDeg > 0)
    {
        // breathing is quasi-periodic: the phase wanders (more with irregularity) and each breath has its own depth
        const double irr = std::min(1.0, std::max(0.0, s.handIrr));
        const double ph = kTwoPi * breathHzQ_ * x.tau + (0.6 + 1.6 * irr) * breathPh_.eval(x.tau);
        const double depth = std::max(0.2, 1 + 0.45 * irr * breathPh_.eval(x.tau * 2.3 + 41.0));
        const double b = depth * (tbm::sin(ph) + 0.2 * tbm::sin(2 * ph + 0.7)) * Gain(breathHzQ_, c_.balPos);
        p.tilt += s.breathDeg * b;
        p.zoom += 0.04 * s.breathDeg * b;
    }
    if (!driftX_.empty())
    {
        double cx, cy;
        Corrections(x.tau, cx, cy);
        p.pan += cx;
        p.tilt += cy;
    }
    Gait(x, p);
    if (s.pulsePct > 0)
    {
        const double rate = std::max(0.05, x.rate);
        // a small radial push that decays (lub, then a weaker dub 0.28 s later), centred off the frame centre:
        // zoom plus a push towards a fixed (seeded) direction, beat strength varies a little
        auto pulse = [](double d) { return d <= 0 ? 0.0 : tbm::exp(-d / 0.14) * (1 - tbm::exp(-d / 0.015)); };
        double v = 0;
        for (int64_t k = FloorI(x.phase); k >= FloorI(x.phase) - 2; --k)
        {
            const double d = (x.phase - (double)k) / rate;
            const double bs = 1 + 0.12 * Gauss(seed_, 801, loopPhase_ > 0 ? Mod(k, (int64_t)std::floor(loopPhase_ + 0.5)) : k);
            v += bs * (pulse(d) + 0.5 * pulse(d - 0.28));
        }
        v /= 0.75;
        const double a = kTwoPi * Uni(seed_, 802, 0);
        p.zoom += 0.6 * s.pulsePct * v * Gain(rate * 4, c_.balZoom);
        p.tx += s.pulsePct * 2.5 * tbm::cos(a) * v * Gain(rate * 4, c_.balPos);
        p.ty += s.pulsePct * 2.5 * tbm::sin(a) * v * Gain(rate * 4, c_.balPos);
    }
    // earthquake envelope (quakeEnv): after each trigger a P phase (rattle only) then the S phase (large slow
    // motion) that decays with the decay control; quakeEnv = 1 means vibration / large motion only come from it
    double envV = 1, envB = 1;
    if (s.quakeEnv > 0)
    {
        const double Tps = 1.8, D = s.decay;
        double P = 0, Sw = 0;
        for (const Trigger& tg : trig)
        {
            const double d = x.tau - tg.tau;
            if (d < 0)
                continue;
            auto rise = [](double u) { return u <= 0 ? 0.0 : u >= 1 ? 1.0 : u * u * (3 - 2 * u); };
            P += tg.strength * rise(d / 0.15) * (d < Tps ? 1.0 : tbm::exp(-(d - Tps) / 1.0));
            if (d > Tps)
                Sw += tg.strength * rise((d - Tps) / 0.6) * (D > 0 ? tbm::exp(-(d - Tps) / D) : 1.0);
        }
        const double q = std::min(1.0, s.quakeEnv);
        envV = (1 - q) + q * (0.7 * P + 0.5 * Sw);
        envB = (1 - q) + q * Sw;
    }
    if (!vibX_.empty())
    {
        const double m0 = m;
        const double m = m0 * envV;
        p.tx += s.vibPx * m * vibX_.eval(x.tau);
        p.ty += s.vibPx * 1.3 * m * vibY_.eval(x.tau);
        p.pan += s.vibPx * 0.004 * m * vibY_.eval(x.tau + 0.37);
        p.tilt += s.vibPx * 0.004 * m * vibX_.eval(x.tau + 0.71);
        p.roll += s.vibPx * 0.01 * m * vibR_.eval(x.tau);
    }
    Bumps(x.tau, p);
    if (!bigX_.empty())
    {
        const double m0 = m;
        const double m = m0 * envB;
        p.tx += s.bigPx * m * bigX_.eval(x.tau);
        p.ty += s.bigPx * 0.8 * m * bigY_.eval(x.tau);
        p.roll += s.bigRot * m * bigR_.eval(x.tau);
        p.zoom += s.bigZoom * m * bigZ_.eval(x.tau);
    }
    Jolts(x.tau, p);
    Film(x, p);
    Impact(x, trig, p);
    return p;
}

Pose Model::Eval(const Ctx& x, const std::vector<Trigger>& trig, double amountNow, const double mulNow[4]) const
{
    Pose r = Raw(x, trig);
    const int axMul[6] = {0, 1, 2, 0, 1, 3};
    for (int i = 0; i < 6; ++i)
        r.set(i, r.get(i) * amountNow * mulNow[axMul[i]]);
    return r;
}

double ShakeRate(double fps, const Shape& s)
{
    fps = fps > 0 ? fps : 24;
    switch (s.frRate)
    {
    case FR_24: return 24;
    case FR_18: return 18;
    case FR_16: return 16;
    case FR_12: return 12;
    case FR_8: return 8;
    case FR_HOLD: return fps / std::max(1, s.holdN);
    default: return fps;
    }
}

double QuantizeTime(double t, double fps, const Shape& s, int64_t* grid, double* gridRate, bool* isComp)
{
    fps = fps > 0 ? fps : 24;
    const double r = ShakeRate(fps, s);
    const bool comp = s.frRate == FR_COMP || (s.frRate == FR_HOLD && s.holdN <= 1);
    // a tiny tolerance (1e-4 of a comp frame) so frame times like k/24 land on the right sample
    const int64_t g = (int64_t)std::floor(t * r + 1e-4 * r / fps);
    if (grid)
        *grid = g;
    if (gridRate)
        *gridRate = r;
    if (isComp)
        *isComp = comp;
    return comp ? t : (double)g / r;
}

// ------------------------------------------------------------------ TimeMap
void TimeMap::Build(const Curve& speed, const Curve& cad, double t0, double t1, double fps, int maxSamples)
{
    n_.clear();
    samples_ = 0;
    t0_ = t0;
    constS_ = !speed.animated;
    constC_ = !cad.animated;
    s0_ = constS_ ? speed.constant : 1;
    c0_ = constC_ ? cad.constant : 1;
    if (constS_ && constC_)
        return;
    if (!(t1 > t0))
        t1 = t0 + 1.0 / std::max(1.0, fps);
    std::vector<double> b{t0, t1};
    bool noKeyAnim = false;
    for (const Curve* c : {&speed, &cad})
    {
        if (!c->animated)
            continue;
        if (c->keys.empty())
            noKeyAnim = true;
        for (double k : c->keys)
            if (k > t0 && k < t1)
                b.push_back(k);
    }
    std::sort(b.begin(), b.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    double h = noKeyAnim ? std::max(1.0 / std::max(1.0, fps), (t1 - t0) / 1024.0) : 0.04;
    int total = 0;
    for (size_t i = 0; i + 1 < b.size(); ++i)
        total += std::max(1, (int)std::ceil((b[i + 1] - b[i]) / h));
    if (total * 2 + (int)b.size() > maxSamples)
        h *= (double)(total * 2 + (int)b.size()) / maxSamples;
    auto S = [&](double t) { return constS_ ? s0_ : speed.at(t); };
    auto C = [&](double t) { return constC_ ? c0_ : cad.at(t); };
    double tau = 0, ph = 0;
    for (size_t i = 0; i + 1 < b.size(); ++i)
    {
        const double a = b[i], e = b[i + 1];
        const int n = std::min(512, std::max(1, (int)std::ceil((e - a) / h)));
        for (int j = 0; j < n; ++j)
        {
            const double x0 = a + (e - a) * j / n, x1 = a + (e - a) * (j + 1) / n, xm = 0.5 * (x0 + x1);
            // the last interval before a key: the left limit at the key is extrapolated from the interval
            // (a hold key jumps exactly there; sampling "just before" is lost in the host's time quantisation)
            const bool atKey = (j + 1 == n) && (i + 2 < b.size());
            const double sa = S(x0), sm = S(xm), ca = C(x0), cm = C(xm);
            const double sb = atKey ? 2 * sm - sa : S(x1), cb = atKey ? 2 * cm - ca : C(x1);
            samples_ += atKey ? 4 : 6;
            Node nd;
            nd.t = x0;
            nd.s = sa;
            nd.c = ca;
            nd.tau = tau;
            nd.ph = ph;
            n_.push_back(nd);
            tau += (x1 - x0) / 6 * (sa + 4 * sm + sb);
            ph += (x1 - x0) / 6 * (sa * ca + 4 * sm * cm + sb * cb);
            // store the end node (left-limit values) as an interval end
            Node ne;
            ne.t = x1;
            ne.s = sb;
            ne.c = cb;
            ne.tau = tau;
            ne.ph = ph;
            n_.push_back(ne);
        }
    }
}

size_t TimeMap::Find(double t) const
{
    // nodes come in pairs (start, end) per interval
    size_t lo = 0, hi = n_.size() / 2;
    while (hi - lo > 1)
    {
        const size_t mid = (lo + hi) / 2;
        if (n_[2 * mid].t <= t)
            lo = mid;
        else
            hi = mid;
    }
    return 2 * lo;
}

static double Hermite(double x0, double x1, double y0, double y1, double d0, double d1, double x)
{
    const double h = x1 - x0;
    if (!(h > 0))
        return y0;
    const double u = (x - x0) / h, u2 = u * u, u3 = u2 * u;
    return (2 * u3 - 3 * u2 + 1) * y0 + (u3 - 2 * u2 + u) * h * d0 + (-2 * u3 + 3 * u2) * y1 + (u3 - u2) * h * d1;
}

double TimeMap::Tau(double t) const
{
    if (n_.empty())
        return s0_ * (t - t0_);
    if (t <= n_.front().t)
        return n_.front().s * (t - n_.front().t);
    if (t >= n_.back().t)
        return n_.back().tau + n_.back().s * (t - n_.back().t);
    const size_t i = Find(t);
    const Node &a = n_[i], &b = n_[i + 1];
    return Hermite(a.t, b.t, a.tau, b.tau, a.s, b.s, t);
}
double TimeMap::Phase(double t) const
{
    if (n_.empty())
        return s0_ * c0_ * (t - t0_);
    if (t <= n_.front().t)
        return n_.front().s * n_.front().c * (t - n_.front().t);
    if (t >= n_.back().t)
        return n_.back().ph + n_.back().s * n_.back().c * (t - n_.back().t);
    const size_t i = Find(t);
    const Node &a = n_[i], &b = n_[i + 1];
    return Hermite(a.t, b.t, a.ph, b.ph, a.s * a.c, b.s * b.c, t);
}
double TimeMap::Speed(double t) const
{
    if (n_.empty())
        return s0_;
    if (t <= n_.front().t)
        return n_.front().s;
    if (t >= n_.back().t)
        return n_.back().s;
    const size_t i = Find(t);
    const Node &a = n_[i], &b = n_[i + 1];
    const double u = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0;
    return a.s + (b.s - a.s) * u;
}
double TimeMap::MaxSpeed() const
{
    if (n_.empty())
        return s0_;
    double m = 0;
    for (const Node& nd : n_)
        m = std::max(m, nd.s);
    return m;
}

double TimeMap::Cadence(double t) const
{
    if (n_.empty())
        return c0_;
    if (t <= n_.front().t)
        return n_.front().c;
    if (t >= n_.back().t)
        return n_.back().c;
    const size_t i = Find(t);
    const Node &a = n_[i], &b = n_[i + 1];
    const double u = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0;
    return a.c + (b.c - a.c) * u;
}

}  // namespace tb
