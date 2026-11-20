#!/bin/sh
# mac_port: builds the GPU static library for macOS as ONE universal archive: <outdir>/libskgpu.a (arm64 + x86_64).
# UNTESTED (written on Windows, where the Apple Rust targets are not installed).
#   rustup target add aarch64-apple-darwin x86_64-apple-darwin      (once, on the Mac)
#   sh gpu/build_gpu_mac.sh <outdir>
# Cargo.lock is committed (unpatched); crates.io must be reachable once (gpu/CRATES_macos.txt lists the crates).
# With the Metal no-fast-math patch: run sh gpu/patches/apply_wgpu_hal_patch.sh FIRST (it refreshes Cargo.lock for
# the patched path copy); --locked below then accepts it.
set -eu
cd "$(dirname "$0")"
out=${1:-target/universal}
mkdir -p "$out"
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
for t in aarch64-apple-darwin x86_64-apple-darwin; do
  cargo rustc --release --locked --lib --crate-type staticlib --target "$t" --target-dir target/mac
done
lipo -create target/mac/aarch64-apple-darwin/release/libskgpu.a target/mac/x86_64-apple-darwin/release/libskgpu.a -output "$out/libskgpu.a"
lipo -info "$out/libskgpu.a"
# the system libraries / frameworks the archive needs (add any missing one to CMakeLists.txt SK_WITH_GPU)
cargo rustc --release --locked --lib --crate-type staticlib --target aarch64-apple-darwin --target-dir target/mac -- --print native-static-libs 2>&1 | grep -i "native-static-libs" || true
