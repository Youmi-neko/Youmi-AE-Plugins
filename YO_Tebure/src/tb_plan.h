// Tebure - per-frame planning shared by the plug-in, the fake host's reference checks and the CLI:
// shot time, loop wrap, shake-frame quantisation, speed / rhythm integration, triggers, motion-blur sub-samples,
// soft limit and pixel snapping. Pure function of its inputs (no host calls).
#pragma once
#include "tb_model.h"

#include <vector>

namespace tb {

struct Hit
{
    double t = 0;        // layer time (s)
    double strength = 1; // 1 = 100 %
};

struct PlanInputs
{
    Shape shape;
    Controls ctl;          // seed + balances (amount / multipliers come from the fields below)
    Curve speed, cadence;  // cadence: constant shape.cadence unless the rhythm control is keyframed (Custom)
    double amountNow = 1, amountMax = 1;
    double mulNow[4] = {1, 1, 1, 1}, mulMax[4] = {1, 1, 1, 1};
    std::vector<Hit> hits; // impulse triggers (layer time); shape.firstHit is added by MakePlan
    double trigMax = 1;
    double t = 0;          // layer time of the frame (s)
    double t0 = 0;         // shot start (layer time, s)
    double fps = 24;       // comp frame rate
    double loop = 0;       // s, 0 = off
    double shutterDeg = 0; // motion blur shutter angle, 0 = off
    int maxSamples = 12;   // motion blur samples (upper bound)
    bool snap = false;     // translations on whole full-res pixels
    double widthRef = 1920;// full-res reference width (px) - for the px estimates of the blur and snapping
    int renderVersion = 3; // 1 = saved by 0.1/0.2 (their rules: 0.2 blur density, no band limit); 2 = saved by 0.3.0
                           // (band limit, 0.3.0's sparser blur rule); 3 = 0.3.1+ (band limit, 0.2 blur density)
    double envScale = 1;   // 0..1: the amplitude envelope at this frame / the shot maximum (auto zoom, soft limit)
    // graph view: poses over [t - graphBack, t + graphAhead]
    int graphSamples = 0;
    double graphBack = 2.5, graphAhead = 2.5;
};

struct PlanOutput
{
    std::vector<Pose> poses;   // motion-blur sub-samples (1 = no blur), equal weights
    Pose bound;                // hard bound per axis (soft limiter) - constant over the shot
    Pose boundAuto;            // the same without whip events: what the auto scale hides
    Pose center;               // the pose at the frame time itself (graph cursor)
    std::vector<double> graphT;
    std::vector<Pose> graph;   // displayed path over the graph window
    double shotTime = 0;       // t - t0 (wrapped by the loop)
    int64_t grid = 0;
    int timeSamples = 0;       // parameter samples used by the time map (cost indicator)
};

PlanOutput MakePlan(const PlanInputs& in);

// px a pose difference moves the frame (rough, for the adaptive blur sample count)
double PoseDistancePx(const Pose& a, const Pose& b, double widthRef, double focal);

}  // namespace tb
