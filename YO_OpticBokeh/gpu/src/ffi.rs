//! repr(C) mirrors of include/obgpu.h (keep in sync; sizes are asserted by the harness).
#![allow(non_camel_case_types)]

use std::os::raw::c_int;

pub const OBGPU_OK: c_int = 0;
pub const OBGPU_BAD_ARGS: c_int = 1;
pub const OBGPU_ABORTED: c_int = 2;
pub const OBGPU_NO_ADAPTER: c_int = 3;
pub const OBGPU_DEVICE_LOST: c_int = 4;
pub const OBGPU_OUT_OF_VRAM: c_int = 5;
pub const OBGPU_TIMEOUT: c_int = 6;
pub const OBGPU_BUSY: c_int = 7;
pub const OBGPU_HOST_OOM: c_int = 8;
pub const OBGPU_UNSUPPORTED: c_int = 9;
pub const OBGPU_INTERNAL: c_int = 10;
pub const OBGPU_SELFTEST_FAILED: c_int = 11;

pub const BACKEND_DX12: u32 = 1;
pub const BACKEND_VULKAN: u32 = 2;
pub const BACKEND_METAL: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct obgpu_config {
    pub struct_size: u32,
    pub backends: u32,
    pub allow_software: i32,
    pub force_no_adapter: i32,
    pub vram_budget_bytes: u64,
    pub timeout_ms: u32,
    pub wait_ms: u32,
    pub max_jobs: i32,
    pub verbose: i32,
}

pub const MAX_LAYERS: usize = 32;
pub const TIERS: usize = 8;
pub const MAX_LEVELS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct obgpu_profile {
    pub nl: i32,
    pub share: i32,
    pub share_from: i32,
    pub _pad: i32,
    pub s2: f64,
    pub s2out: f64,
    pub s2in: f64,
    pub _pad2: f64,
    pub ls: [f64; MAX_LAYERS],
    pub lb: [f64; MAX_LAYERS],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_level {
    pub f: i32,
    pub w: i32,
    pub h: i32,
    pub gox: i32,
    pub goy: i32,
    pub max_reach: i32,
    pub row_lo: i32,
    pub row_hi: i32,
}

pub const DESC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct obgpu_blur_desc {
    pub struct_size: u32,
    pub version: u32,
    pub w: i32,
    pub h: i32,
    pub nk: i32,
    pub n_tiers: i32,
    pub n_levels: i32,
    pub cat: i32,
    pub cat_n: i32,
    pub clamp_neg: i32,
    pub fringe_share: i32,
    pub need_el: i32,
    pub max_reach: i32,
    pub row_begin: i32,
    pub row_end: i32,
    pub ap_n: i32,
    pub circ_n: i32,
    pub _pad0: i32,
    pub ax: f64,
    pub ay: f64,
    pub xext: f64,
    pub yext: f64,
    pub cat_k: f64,
    pub cat_lim: f64,
    pub cat_inv: f64,
    pub level_base: f64,
    pub fcx: f64,
    pub fcy: f64,
    pub half_diag: f64,
    pub pa: f64,
    pub max_radius: f64,
    pub smax_g: f64,
    pub ap_ymin: f64,
    pub ap_ymax: f64,
    pub ap_inv_step: f64,
    pub ap_area: f64,
    pub circ_ymin: f64,
    pub circ_ymax: f64,
    pub circ_inv_step: f64,
    pub circ_area: f64,
    pub ks: [[f64; 3]; 2],
    pub prof: [[obgpu_profile; TIERS]; 2],
    pub levels: [obgpu_level; MAX_LEVELS],
    pub ap_table: *const f64,
    pub circ_table: *const f64,
    pub cat_a: *const f64,
}

#[repr(C)]
pub struct obgpu_input {
    pub struct_size: u32,
    pub rgba: *const f32,
    pub radius: *const f64,
    pub side: *const i8,
    pub el: *const f32,
}

#[repr(C)]
pub struct obgpu_output {
    pub struct_size: u32,
    pub rgba: *mut f32,
    pub rgba_stride: i32,
    pub el: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_stats {
    pub total_ms: f64,
    pub convert_ms: f64,
    pub upload_ms: f64,
    pub gpu_ms: f64,
    pub readback_ms: f64,
    pub wait_ms: f64,
    pub vram_bytes: u64,
    pub bands: i32,
    pub submissions: i32,
}

pub type obgpu_abort_fn = Option<unsafe extern "C" fn(user: *mut std::ffi::c_void) -> c_int>;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_selftest_report {
    pub struct_size: u32,
    pub pass: i32,
    pub cases: i32,
    pub deterministic: i32,
    pub failed_case: i32,
    pub max_rel: f64,
    pub mean_abs: f64,
    pub ms: f64,
    /// ABI 1.5: GPU busy ms per work unit of the self-test frame cases (timestamps; wall if unsupported) - the
    /// router's cold-start seed (an upper bound: tiny frames carry the fixed per-pass cost)
    pub seed_ms_per_wu: f64,
    /// ABI 1.5: wall and GPU busy ms of the (warm) self-test frame cases: wall / busy = how much the GPU made the
    /// self-test wait (other applications) at start-up
    pub seed_wall_ms: f64,
    pub seed_busy_ms: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_counters {
    pub struct_size: u32,
    pub route_gpu: i32,
    pub gpu_jobs: u64,
    pub gpu_failed: u64,
    pub cpu_routed: u64,
    pub probes: u64,
    pub cpu_reports: u64,
    pub ema_gpu_ms_per_wu: f64,
    pub ema_cpu_ms_per_wu: f64,
    pub last_wall_ms: f64,
    pub last_gpu_busy_ms: f64,
    pub ema_busy_fraction: f64,
    pub active_jobs: i32,
    pub waiting_jobs: i32,
    pub timestamps: i32,
    pub deadline_aborts: u64,
    pub timeouts: u64,
    pub last_prep_bands: i32,
    pub last_prep_rows_factor: f64,
    pub vram_scale: f64,
    pub seed_ms_per_wu: f64,
    // ---- ABI 1.5.2 (appended; struct_size-gated): why GPU jobs returned OBGPU_BUSY before getting a job slot
    /// the slot wait passed its limit (wait_ms, or the queue moved far slower than the GPU's own pace)
    pub busy_wait_limit: u64,
    /// while waiting: the GPU could no longer finish before the routing deadline (1.25 x the CPU time)
    pub busy_queue: u64,
    /// while waiting: another job hit the deadline / a submission timed out (saturated or stuck GPU)
    pub busy_saturated: u64,
}

/// ABI 1.5.2: what happened to the last GPU job of the calling thread (obgpu_last_job; diagnostics only)
pub const OBGPU_BUSY_NONE: i32 = 0;
pub const OBGPU_BUSY_WAIT_LIMIT: i32 = 1;
pub const OBGPU_BUSY_QUEUE: i32 = 2;
pub const OBGPU_BUSY_SATURATED: i32 = 3;
pub const OBGPU_BUSY_DEADLINE: i32 = 4;
pub const OBGPU_BUSY_COLD_DEADLINE: i32 = 5;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct obgpu_job_info {
    pub struct_size: u32,
    /// status the call returned
    pub status: i32,
    /// OBGPU_BUSY_* when status == OBGPU_BUSY (else 0)
    pub busy_reason: i32,
    /// jobs running + waiting when this call arrived (itself excluded)
    pub ahead: i32,
    /// prepare row bands of a frame job (0 = none / not a frame job)
    pub prep_bands: i32,
    pub _pad: i32,
    /// time waiting for a job slot, time from getting the slot to the end (0 = never got one), whole call
    pub slot_wait_ms: f64,
    pub run_ms: f64,
    pub total_ms: f64,
    /// the routing deadline in force at the end (ms; NaN = none) and whether it counts the job's own time
    /// (cold start, since it got its slot) rather than the time since the call
    pub deadline_ms: f64,
    pub deadline_own: i32,
    pub _pad2: i32,
    /// the router's estimates for this job at the end: GPU own time, CPU time (ms; NaN = unknown)
    pub est_gpu_ms: f64,
    pub est_cpu_ms: f64,
    /// VRAM budget planned for the job (bytes; 0 = never got a slot)
    pub budget_bytes: u64,
    pub work_units: f64,
}
impl Default for obgpu_job_info {
    fn default() -> Self {
        obgpu_job_info {
            struct_size: std::mem::size_of::<obgpu_job_info>() as u32,
            status: 0,
            busy_reason: 0,
            ahead: 0,
            prep_bands: 0,
            _pad: 0,
            slot_wait_ms: 0.0,
            run_ms: 0.0,
            total_ms: 0.0,
            deadline_ms: f64::NAN,
            deadline_own: 0,
            _pad2: 0,
            est_gpu_ms: f64::NAN,
            est_cpu_ms: f64::NAN,
            budget_bytes: 0,
            work_units: 0.0,
        }
    }
}

// ---- ABI 1.3: post stages
pub const OBGPU_STAGE_POST: u32 = 4;
pub const OBGPU_POST_VERSION: u32 = 1;
pub const OBGPU_POST_GLOWVIG: u32 = 1;
pub const OBGPU_POST_HALATION: u32 = 2;
pub const OBGPU_POST_LUMAB: u32 = 4;
pub const OBGPU_POST_ROLLOFF: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_field_op {
    pub kind: i32,
    pub r: i32,
    pub f: i32,
    pub gx: i32,
    pub gy: i32,
    pub gw: i32,
    pub gh: i32,
    pub rb0: i32,
    pub rb1: i32,
    pub sigma: f32,
    pub t: f32,
    pub _pad: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_post_desc {
    pub struct_size: u32,
    pub version: u32,
    pub w: i32,
    pub h: i32,
    pub stages: u32,
    pub roll_candidate: i32,
    pub glow_amt: f32,
    pub vig: f32,
    pub glow_f: i32,
    pub glow_gw: i32,
    pub glow_gh: i32,
    pub glow_gox: i32,
    pub glow_goy: i32,
    pub glow_rb: i32,
    pub fcx: f32,
    pub fcy: f32,
    pub half_diag: f32,
    pub pa: f32,
    pub hal_amt: f32,
    pub hal_floor: f32,
    pub hal_relk: f32,
    pub hal_soft: f32,
    pub hal_rim: f32,
    pub hal_tint: [f32; 3],
    pub hal_scene: obgpu_field_op,
    pub hal_tight: obgpu_field_op,
    pub hal_wide: obgpu_field_op,
    pub la_mode: i32,
    pub la_amt: f32,
    pub la_thr: f32,
    pub la_curve: f32,
    pub la_hs: f32,
    pub la_hi: [f32; 3],
    pub la_field: obgpu_field_op,
    pub roll_v2: i32,
    pub roll_knee: f32,
    // ---- ABI 1.4 (version 2)
    pub la34: i32,
    pub dr_s1: i32,
    pub dr_s2: i32,
    pub dr_v2: i32,
    pub dr_sp1: i32,
    pub dr_sp2: i32,
    pub dr_thr: f32,
    pub dr_curve: f32,
    pub dr_pres: f32,
    pub dr_kstop: f32,
    pub dr_len: f32,
    pub dr_sm2: f32,
    pub dr_lencap: f32,
    pub dr_k: [f32; 3],
    pub dr_soft: obgpu_field_op,
    pub dr_dir: obgpu_field_op,
    pub rim_s1: i32,
    pub rim_s2: i32,
    pub rim_rn: i32,
    pub rim_rc: i32,
    pub rim_fringe: i32,
    pub rim_bright_side: i32,
    pub rim_mixb_on: i32,
    pub _pad2: i32,
    pub rim_amt: f32,
    pub rim_thr: f32,
    pub rim_curve: f32,
    pub rim_hs: f32,
    pub rim_ex: f32,
    pub rim_mixb: f32,
    pub rim_comp: f32,
    pub rim_hi: [f32; 3],
    pub rim_reg: obgpu_field_op,
    pub rim_fb: obgpu_field_op,
    pub rim_fd: obgpu_field_op,
    pub rim_sig: [obgpu_field_op; 3],
    pub rim_s0: obgpu_field_op,
}
pub const OBGPU_POST_VERSION2: u32 = 2;
pub const OBGPU_POST_LA34: u32 = 16;
/// size of the ABI 1.3 (version 1) post desc
pub const POST_DESC_V1_SIZE: usize = std::mem::offset_of!(obgpu_post_desc, la34);

pub const OBGPU_STAGE_PREPARE: u32 = 1;
pub const OBGPU_STAGE_BLUR: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct obgpu_prepare_desc {
    pub struct_size: u32,
    pub version: u32,
    pub w: i32,
    pub h: i32,
    pub has_map: i32,
    pub invert_map: i32,
    pub hl_on: i32,
    pub render_version: i32,
    pub focus: f32,
    pub map_scale: f32,
    pub map_gamma: f32,
    pub max_radius: f32,
    pub hl_gain: f32,
    pub hl_threshold: f32,
    pub boost: f32,
    pub knee: f32,
    pub curve: f32,
    pub sel: f32,
    pub hsat: f32,
    pub ps: f32,
    pub adapt_gain: f32,
    pub knee_box_r: i32,
    pub wp_on: i32,
    pub wp_amt: f32,
    pub wp_f: i32,
    pub wp_gx: i32,
    pub wp_gy: i32,
    pub wp_gw: i32,
    pub wp_gh: i32,
    pub wp_rb: i32,
    pub wp_rel: f32,
    pub _pad0: f32,
    pub wp_tau: [f32; 10],
    pub det_on: i32,
    pub det_n: i32,
    pub dmin_on: i32,
    pub dmin_n: i32,
    pub det_rf: [f32; 8],
    pub det_w: [f32; 8],
    pub dmin_rf: [f32; 8],
    pub dmin_w: [f32; 8],
    pub dmin_ramp: f32,
    pub dots: f32,
    pub dots_on: i32,
    pub margin: i32,
    pub dots_r: [i32; 4],
}

#[repr(C)]
pub struct obgpu_frame_input {
    pub struct_size: u32,
    pub src: *const f32,
    pub src_stride: i32,
    pub map: *const f32,
    pub map_stride: i32,
    pub crop_x: i32,
    pub crop_y: i32,
    pub crop_w: i32,
    pub crop_h: i32,
}

#[repr(C)]
pub struct obgpu_frame_output {
    pub struct_size: u32,
    pub rgba: *mut f32,
    pub rgba_stride: i32,
    pub el: *mut f32,
    pub prepared_rgba: *mut f32,
    pub roll_relevant: i32,
    pub stages: u32,
    pub prepared_el: *mut f32,
}