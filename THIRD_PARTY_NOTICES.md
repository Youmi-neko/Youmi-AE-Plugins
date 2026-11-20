# Third-Party Notices

This repository (`Youmi-AE-Plugins`) is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE) for the full text.

The plug-ins in this repository depend on or link against the following third-party open-source components and platform SDKs when compiled:

---

## 1. Rust & `wgpu` Ecosystem Crates (GPU Acceleration)

Used by `YO_OpticBokeh/gpu/` (shared by `YO_OpticBokeh` and `YO_OpticBokehKantan`), `YO_MapBilinearBlur_ja/gpu/`, and `YO_Shitsukan/gpu/` for cross-platform GPU compute acceleration (Vulkan / DirectX 12 on Windows, Metal on macOS).

- **`wgpu` / `wgpu-core` / `wgpu-hal` / `naga` / `wgpu-types`** — Dual-licensed under **MIT** or **Apache-2.0** (https://github.com/gfx-rs/wgpu)
- **`bytemuck`**, **`pollster`**, **`ash`**, **`windows`**, **`objc2`** and transitive crates pinned in each plugin's `gpu/Cargo.lock` — Licensed under permissive open-source licenses (**MIT**, **Apache-2.0**, **BSD-2-Clause**, **BSD-3-Clause**, **Zlib**, **ISC**, **Unicode-3.0**).

The exact resolved dependency graph and versions are recorded in:
- `YO_OpticBokeh/gpu/Cargo.lock`
- `YO_MapBilinearBlur_ja/gpu/Cargo.lock`
- `YO_Shitsukan/gpu/Cargo.lock`

---

## 2. Adobe After Effects SDK

- **Adobe After Effects SDK** headers and build tools (`PiPLtool`, `Rez` headers) are **not included** in this repository.
- To build the plug-ins from source, obtain the After Effects SDK directly from Adobe (https://developer.adobe.com/after-effects/) and pass its `Examples/` directory to CMake / `build_win.ps1`.
- Building against the After Effects SDK is subject to Adobe's After Effects SDK License Agreement.

---

## 3. Compiler & Platform Runtimes

- **Microsoft Visual C++ Static Runtime (`/MT`)** on Windows builds — subject to the Microsoft Visual Studio License Terms.
- **Apple macOS System Frameworks** (`CoreFoundation`, `Foundation`, `Metal`, `QuartzCore`, `CoreGraphics`, `IOSurface`, `IOKit`) — dynamically linked at runtime on macOS.
