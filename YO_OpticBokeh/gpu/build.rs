// build.rs - selftest.bin freshness guard. The embedded self-test data (src/selftest.bin) is the CPU engine's
// result for a few tiny cases; if the engine or the desc shims change, the data must be regenerated with
// `build\gpu\obgpu_harness.exe mkselftest`, or customers' adapters would be judged against a stale reference.
// The file stores an FNV-1a hash of the sources below (same list and order as the harness); a mismatch fails
// the build. OBGPU_ALLOW_STALE_SELFTEST=1 turns the error into a warning (only to build the harness that
// regenerates it).
use std::path::Path;

fn fnv(mut h: u64, p: &Path) -> u64 {
    match std::fs::read(p) {
        Ok(b) => {
            for x in b {
                h ^= x as u64;
                h = h.wrapping_mul(1099511628211);
            }
            h
        }
        Err(_) => h ^ 0xdeadbeef,
    }
}

fn main() {
    let gpu = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = gpu.parent().unwrap();
    // mac port package: the engine lives in <root>/src/core (the Windows project tree: <root>/core). The hash covers the same six files in the same order (the
    // stored hash of the v2.4.2 selftest.bin stays valid: file CONTENTS are hashed, not paths).
    let engine = if root.join("src/core/lenscore.cpp").exists() { root.join("src") } else { root.to_path_buf() };
    let srcs = ["core/lenscore.cpp", "core/lenscore.h", "core/lens_presets.h", "gpu/cpp/obgpu_desc.cpp", "gpu/cpp/obgpu_prepdesc.cpp", "gpu/include/obgpu_lenscore.h"];
    let mut h: u64 = 1469598103934665603;
    for s in srcs {
        let p = if s.starts_with("core/") { engine.join(s) } else { root.join(s) };
        println!("cargo:rerun-if-changed={}", p.display());
        h = fnv(h, &p);
    }
    let bin = gpu.join("src/selftest.bin");
    println!("cargo:rerun-if-changed={}", bin.display());
    println!("cargo:rerun-if-env-changed=OBGPU_ALLOW_STALE_SELFTEST");
    let stored = std::fs::read(&bin).ok().filter(|b| b.len() >= 16 && &b[0..4] == b"OBST" && b[4..8] == 2u32.to_le_bytes()).map(|b| u64::from_le_bytes(b[8..16].try_into().unwrap()));
    let ok = stored == Some(h);
    if !ok {
        let msg = format!(
            "src/selftest.bin is stale or missing (stored {:?}, sources {:016x}): run build\\gpu\\obgpu_harness.exe mkselftest (build the harness with OBGPU_ALLOW_STALE_SELFTEST=1 first)",
            stored.map(|v| format!("{v:016x}")),
            h
        );
        if std::env::var("OBGPU_ALLOW_STALE_SELFTEST").is_ok() {
            println!("cargo:warning={msg}");
        } else {
            panic!("{msg}");
        }
    }
}
