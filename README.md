# Youmi AE Plugins

Adobe After Effects 向けのネイティブエフェクトプラグイン集（Windows x64 / macOS Universal `arm64 + x86_64` 対応）です。  
エフェクトパネルの **`Youmi`** カテゴリに追加されます。

A collection of native Adobe After Effects effect plug-ins for **Windows (x64)** and **macOS (Universal: Apple Silicon + Intel)**.  
All effects appear under the **`Youmi`** effect category in After Effects.

---

## 収録プラグイン / Included Plug-ins

| プラグイン / Plug-in | 表示名 / Effect Name | 概要 / Description | ディレクトリ / Source |
|---|---|---|---|
| **YO_Tebure** | `YO Tebure Shake` | 物理ベースの手ブレ・カメラシェイクエフェクト（プリセット選択・モーション解析・モーションブラー・リニアブレンド対応） / Physics-based handheld camera shake effect with presets, motion blur, and linear-light blending | [`YO_Tebure/`](YO_Tebure/) |
| **YO_OpticBokehKantan** | `YO Optic Bokeh Kantan` | 光学レンズボケ・収差・フィルムグレインを直感的に操作できる簡易版エフェクト（Rust/wgpu GPU 高速化対応） / Streamlined optical lens bokeh, aberration, and film grain effect with Rust/wgpu GPU acceleration | [`YO_OpticBokehKantan/`](YO_OpticBokehKantan/) |
| **YO_MapBilinearBlur_ja** | `YO Map Bilinear Blur` | 深度マップ（グレースケールマップ）に応じて滑らかにボケ量を制御するピラミッドブラー（日本語 UI・GPU 対応） / Depth-map-driven pyramid bilinear blur with Japanese UI and GPU acceleration | [`YO_MapBilinearBlur_ja/`](YO_MapBilinearBlur_ja/) |
| **YO_Shitsukan** | `YO Shitsukan` | 質感・線画保持ブラー・水彩風タッチを与えるテクスチャエフェクト（Rust/wgpu GPU 高速化対応） / Texture, line-preserving blur, and watercolor stylization effect with Rust/wgpu GPU acceleration | [`YO_Shitsukan/`](YO_Shitsukan/) |

---

## インストール方法 / Installation

ビルド済みバイナリは [Releases](../../releases) からダウンロードできます。  
Prebuilt binaries are available on the [Releases](../../releases) page.

### Windows (x64)
1. `Youmi-AE-Plugins-v*-Windows-x64.zip` を解凍します。
2. 中に含まれる `.aex` ファイルを以下のフォルダへコピーします：
   ```text
   C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\
   ```
3. After Effects を再起動します。

### macOS (Universal: Apple Silicon & Intel)
1. `Youmi-AE-Plugins-v*-macOS-Universal.zip` を解凍します（中に各プラグインの `.plugin.zip` または `.plugin` バンドルが入っています）。
2. 各 `.plugin` バンドルを以下のフォルダへコピーします：
   ```text
   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/
   ```
3. **初回起動時のセキュリティ警告（Gatekeeper）について**:  
   配布バイナリは公証（Notarization）を行っていないため、macOS で読み込めない場合はターミナルで検疫属性を解除してください：
   ```bash
   xattr -dr com.apple.quarantine "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/YO_"*.plugin
   ```
4. After Effects を再起動します。

---

## ソースからのビルド方法 / Building from Source

### 必要なもの / Prerequisites

- **Adobe After Effects SDK**（本リポジトリには含まれません。[Adobe Developer](https://developer.adobe.com/after-effects/) から取得し、`Examples/` フォルダのパスを指定してください）
  - Tested with After Effects SDK 26.5
- **CMake 3.21+** および **Ninja**（または Xcode / Visual Studio）
- **C++17 コンパイラ**:
  - Windows: Visual Studio 2022 (`x64 Native Tools Command Prompt`)
  - macOS: Xcode Command Line Tools (`clang`, `xcrun Rez`, `lipo`)
- **Rust toolchain (`cargo`)** — `YO_OpticBokehKantan`, `YO_MapBilinearBlur_ja` (Win GPU), `YO_Shitsukan` (GPU 有効時) に必要
  - macOS Universal ビルド時は `rustup target add aarch64-apple-darwin x86_64-apple-darwin` を実行してください

---

### 1. `YO_Tebure`

```bash
# Windows (x64 Native Tools Prompt)
cmake -S YO_Tebure -B build/tebure -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK="C:/path/to/AfterEffectsSDK_26.5_win/Examples"
cmake --build build/tebure

# macOS (Universal arm64 + x86_64)
cmake -S YO_Tebure -B build/tebure -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK="/path/to/AfterEffectsSDK_26.5_mac/Examples"
cmake --build build/tebure
```

---

### 2. `YO_OpticBokehKantan`

> **Note**: `-DAE_SDK_DIR` を使用します（GPU なし CPU のみでビルドする場合は `-DOB_WITH_GPU=OFF`）。

```bash
# Windows (x64 Native Tools Prompt)
cmake -S YO_OpticBokehKantan -B build/obk -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK_DIR="C:/path/to/AfterEffectsSDK_26.5_win/Examples"
cmake --build build/obk

# macOS (Universal arm64 + x86_64)
cmake -S YO_OpticBokehKantan -B build/obk -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK_DIR="/path/to/AfterEffectsSDK_26.5_mac/Examples"
cmake --build build/obk
```

---

### 3. `YO_MapBilinearBlur_ja`

```powershell
# Windows (x64 Native Tools PowerShell, GPU 有効)
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo rustc --manifest-path YO_MapBilinearBlur_ja/gpu/Cargo.toml --release --locked --lib --crate-type staticlib --target-dir build/mbb_cargo
cmake -S YO_MapBilinearBlur_ja -B build/mbb -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DAE_SDK="C:/path/to/AfterEffectsSDK_26.5_win/Examples" `
  -DMBB_GPU=ON -DMBB_GPULIB="build/mbb_cargo/release/mbbgpu.lib"
cmake --build build/mbb
```

```bash
# macOS (Universal arm64 + x86_64, CPU エンジン)
cmake -S YO_MapBilinearBlur_ja -B build/mbb -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK="/path/to/AfterEffectsSDK_26.5_mac/Examples" -DMBB_GPU=OFF
cmake --build build/mbb
```

---

### 4. `YO_Shitsukan`

```powershell
# Windows (x64 Native Tools PowerShell, GPU 有効)
.\YO_Shitsukan\build_win.ps1 `
  -PackageDir .\YO_Shitsukan `
  -AeSdkExamples "C:\path\to\AfterEffectsSDK_26.5_win\Examples" `
  -OutDir .\build\shitsukan `
  -WithGpu
```

```bash
# macOS (Universal arm64 + x86_64)
cmake -S YO_Shitsukan -B build/shitsukan -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK="/path/to/AfterEffectsSDK_26.5_mac/Examples" -DSK_WITH_GPU=OFF
cmake --build build/shitsukan
```

---

## ライセンス / License

本リポジトリのソースコードは **GNU General Public License v3.0 (GPL-3.0)** のもとで公開されています。  
詳細は [LICENSE](LICENSE) および [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) をご覧ください。

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** — see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
