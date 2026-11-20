// Tebure (camera shake) - the motion model. No After Effects dependency (shared by the plug-in, the fake host,
// the CLI and the tests). See docs\shake_model.md for the physics and the reasoning.
//
// The camera "pose" is a deterministic function of (shot time, params, seed):
//   pan / tilt / roll (deg, camera rotation, projected exactly as a pinhole rotation homography),
//   tx / ty (px of a 1920 px wide frame, i.e. frame-relative: scaled by width/1920; content shift from camera
//   translation at the framing distance), zoom (%: content scale = exp(zoom/100)).
// Every generator is random-access: sums of sinusoids with hashed frequencies/phases, hashed event lattices
// (operator corrections, road bumps, jolts), integrated rhythm phase (steps / heartbeat), triggered responses.
// Nothing depends on render order, frame rate (except the explicit frame-quantised parts) or threads.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace tb {

// ------------------------------------------------------------------ the preset-written "shape"
enum ImpType { IMP_PHYSICAL = 0, IMP_GAMENDO, IMP_WHIP, IMP_COUNT };
enum JoltMode { JOLT_FREE = 0, JOLT_H, JOLT_V, JOLT_RECT, JOLT_COUNT };
enum FrameRateMode { FR_COMP = 0, FR_24, FR_18, FR_16, FR_12, FR_8, FR_HOLD, FR_COUNT };

struct Shape
{
    // camera / sampling
    double focal = 35;        // mm, 35 mm equivalent (36 mm wide)
    int frRate = FR_COMP;     // shake frame rate
    int holdN = 1;            // FR_HOLD: hold every N comp frames
    // hand-held
    double handDeg = 0;       // RMS sway angle (pan; tilt = handDeg * handYX)
    double handHz = 1;        // sway corner / resonance frequency
    double handQ = 0.7;       // resonance sharpness (0.5 = smooth low-pass, 8 = tripod ringing)
    double handRough = 0;     // -1..1: spectral slope above handHz f^-6 (very smooth) .. f^-4 .. f^-2 (rough)
    double handYHz = 1;       // vertical sway frequency / horizontal
    double handIrr = 0;       // 0..1 irregularity: time warp (arcs of varying length), per-arc size, short pauses
    double handYX = 1;        // vertical / horizontal
    double handRoll = 0.3;    // roll RMS = handDeg * handRoll
    double bodyPx = 0;        // translational body sway (px @1920)
    double bodyHz = 0.5;      // its corner frequency
    double tremorDeg = 0;     // 8..12 Hz physiological tremor RMS
    double breathDeg = 0;     // breathing tilt amplitude
    double breathHz = 0.25;
    double driftDeg = 0;      // slow drift, recentred by the operator
    double corrInt = 3;       // mean interval between recentring corrections (s)
    // walking / rhythm
    double cadence = 1.8;     // steps (or beats) per second - integrated with the speed
    double bouncePx = 0;      // vertical bounce per step
    double swayPx = 0;        // lateral sway (per stride = 2 steps)
    double walkRoll = 0;      // deg, per stride
    double walkPan = 0;       // deg, per stride
    double heelPx = 0;        // heel-strike jolt
    double heelSharp = 0;     // 0 walking .. 1 running (flight phase, sharp landing)
    double gaitVar = 0;       // 0..1 step timing / size jitter, left-right asymmetry, occasional stumble
    double pulsePct = 0;      // heartbeat zoom pulse (lub-dub) on the rhythm
    // vibration / large motion
    double vibPx = 0, vibHz = 25, vibBand = 0.15;
    double bumpPx = 0, bumpRate = 0.5;
    double bigPx = 0, bigHz = 0.3, bigRot = 0, bigZoom = 0;
    double swell = 0;         // 0..1 slow amplitude modulation depth
    double quakeEnv = 0;      // 0..1: vibration / large motion follow a P-then-S envelope after each trigger
    // impacts (triggered)
    int impType = IMP_PHYSICAL;
    double impPx = 0, impRot = 0, impZoom = 0, impHz = 6, impGrit = 0;
    double impAngle = 90;     // deg, 90 = vertical (image y down)
    double impRandom = 0.3;   // direction / amplitude randomness 0..1
    double decay = 0.5;       // s, 0 = sustained
    double firstHit = -1;     // s after the shot start, <0 = none
    double hitEvery = 0;      // s: random extra triggers at about this mean interval, 0 = none
    double shutter = 0;       // motion blur shutter angle written by the preset (deg)
    // jolts / stepwise reframes
    double joltPx = 0, joltInt = 3, joltDur = 0.1, joltSettle = 0.15;
    int joltMode = JOLT_FREE;
    double joltYX = 0.4;
    // film
    double weavePx = 0, jitterPx = 0, jitterYX = 1;
};

// The user-level controls on top of the shape (never written by presets).
struct Controls
{
    double amount = 1;        // 蠑ｷ縺・    double mulX = 1, mulY = 1, mulRot = 1, mulZoom = 1;
    double balPos = 0, balRot = 0, balZoom = 0;  // -1..1 spectral tilt (low <-> high)
    uint32_t seed = 0;
    double bandSpeed = 0;  // >0: the shot's max time speed; with nyquist > 0 the motion pushed above Nyquist is faded
    double nyquist = 0;    // Hz of real time (half the shake sampling rate)
};

struct Pose
{
    double pan = 0, tilt = 0, roll = 0;  // deg
    double tx = 0, ty = 0;               // px @1920 wide
    double zoom = 0;                     // %
    double get(int i) const { return i == 0 ? pan : i == 1 ? tilt : i == 2 ? roll : i == 3 ? tx : i == 4 ? ty : zoom; }
    void set(int i, double v) { (i == 0 ? pan : i == 1 ? tilt : i == 2 ? roll : i == 3 ? tx : i == 4 ? ty : zoom) = v; }
};
Pose Lerp(const Pose& a, const Pose& b, double u);

// A trigger (impulse) at shot-time tau (already speed-mapped) with a strength (1 = full).
struct Trigger
{
    double tau = 0;
    double strength = 1;
    int64_t id = 0;   // stable identity (hash input for the random direction)
    int64_t grid = 0; // shake-frame grid index at the trigger (gamen-do alternation starts there)
};

// Shake-frame sampling ("謠ｺ繧後・繝輔Ξ繝ｼ繝繝ｬ繝ｼ繝・): the instant whose motion is shown at layer/shot time t.
// fps = comp frame rate. Returns the sample time; grid index / rate / whether it is the comp grid.
double QuantizeTime(double t, double fps, const Shape& s, int64_t* grid, double* gridRate, bool* isComp);
double ShakeRate(double fps, const Shape& s);  // samples per second of the grid

// Evaluation context for one instant.
struct Ctx
{
    double tau = 0;        // integrated (speed-scaled) shot time, s
    double phase = 0;      // integrated rhythm phase (steps), = integral of cadence * speed
    double rate = 1.8;     // instantaneous d(phase)/d(tau) (cadence), for heel-strike timing
    int64_t grid = 0;      // index of the shake-frame sample (film frame / held step) since the shot start
    double gridRate = 24;  // samples per second of that grid (frame-quantised parts)
    bool gridIsComp = true;// the grid is the comp frame grid (no explicit shake frame rate)
    double tauFilm = 0;    // tau used for film jitter when gridIsComp (24 fps of tau)
};

// One built model: all random tables for (shape, controls, seed, loop). Build once per render, evaluate many.
class Model
{
public:
    // loopTau > 0: every generator is made exactly periodic in tau with this period; loopGrid = number of
    // grid samples per loop (frame-quantised parts), loopPhase = rhythm phase per loop (rounded to strides).
    void Build(const Shape& s, const Controls& c, double loopTau = 0, int64_t loopGrid = 0, double loopPhase = 0);
    // raw pose (before multipliers / soft limit) - used by tests
    Pose Raw(const Ctx& x, const std::vector<Trigger>& trig) const;
    // final pose: raw * amount(t) * multipliers, then the soft limiter at the hard bound
    Pose Eval(const Ctx& x, const std::vector<Trigger>& trig, double amountNow, const double mulNow[4]) const;
    // the hard bound of |pose[i]| (deg / px / %) for amount <= amountMax, multipliers <= mulMax, trigger
    // strengths <= trigMax (soft limiter guarantees it); multiHit = more than one trigger in the shot (overlaps)
    // forAutoScale: whip events excluded (they are allowed to reveal the edge, which is then repeated)
    Pose Bound(double amountMax, const double mulMax[4], double trigMax, bool multiHit = true, bool forAutoScale = false) const;
    double ResponseLength() const;  // s of tau after a trigger that still matters (inf -> 1e9)
    const Shape& shape() const { return sh_; }

    struct Sines
    {
        std::vector<double> f, a, ph;
        double rmsBound = 0;  // sum of |a| (strict) and RMS = 1 by construction
        double eval(double t) const;
        bool empty() const { return f.empty(); }
    };

private:
    Shape sh_;
    Controls c_;
    double loopTau_ = 0, loopPhase_ = 0;
    int64_t loopGrid_ = 0;
    // continuous processes (unit RMS)
    Sines handX_, handY_, handR_, bodyX_, bodyY_, tremX_, tremY_, breathPh_, driftX_, driftY_;
    Sines warpX_, warpY_, amodX_, amodY_;
    Sines vibX_, vibY_, vibR_, bigX_, bigY_, bigR_, bigZ_, swell_, weaveX_, weaveY_, gaitJit_, gaitAmp_, grit_;
    double gainPos_ = 1, gainRot_ = 1, gainZoom_ = 1;  // balance gains for event parts
    double breathHzQ_ = 0.25;
    double rmsPos_[6] = {0};    // stationary RMS per pose axis (for the bound)
    double peak_[6] = {0};      // deterministic peaks per axis (per unit strength / amount)
    double trigPeak_[6] = {0};  // peaks of one trigger response
    uint64_t seed_ = 0;

    double Swell(double tau) const;
    double bandS_ = 0, bandN_ = 0;
    double Taper(double fr) const;
    double BandGain(double f) const;
    void Corrections(double tau, double& x, double& y) const;
    void Gait(const Ctx& x, Pose& p) const;
    void Bumps(double tau, Pose& p) const;
    void Jolts(double tau, Pose& p) const;
    void Film(const Ctx& x, Pose& p) const;
    void Impact(const Ctx& x, const std::vector<Trigger>& trig, Pose& p) const;
    double CellTime(int64_t k, double cell, uint64_t salt, bool& occupied) const;
};

// ------------------------------------------------------------------ time mapping (speed / rhythm integration)
// A parameter curve as the host gives it: value at a time, keyframe times (empty + !animated = constant).
struct Curve
{
    std::function<double(double)> at;  // seconds (layer time) -> value
    std::vector<double> keys;          // keyframe times (s), sorted
    bool animated = false;             // value differs somewhere (keys or expression)
    double constant = 1;               // value if !animated
};

// tau(t) = integral_{t0}^{t} speed, phase(t) = integral_{t0}^{t} speed * cadence. Built for [t0, t1].
class TimeMap
{
public:
    // fps: comp frames per second (fallback sampling step for curves animated without keyframes)
    void Build(const Curve& speed, const Curve& cadence, double t0, double t1, double fps, int maxSamples = 6144);
    double Tau(double t) const;
    double Phase(double t) const;
    double Speed(double t) const;
    double Cadence(double t) const;
    double MaxSpeed() const;
    int samples() const { return samples_; }

private:
    struct Node { double t, s, c, tau, ph; };
    std::vector<Node> n_;
    bool constS_ = true, constC_ = true;
    double s0_ = 1, c0_ = 1, t0_ = 0;
    int samples_ = 0;
    size_t Find(double t) const;
};

// the soft limiter: |out| < L always, ~identity below 0.6 L
double SoftLimit(double x, double L);

// hashing helpers (exposed for tests)
uint64_t Mix64(uint64_t x);
double Uni(uint64_t seed, uint64_t salt, int64_t i);
double Gauss(uint64_t seed, uint64_t salt, int64_t i);

}  // namespace tb
