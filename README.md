# Youmi AE Plugins

Adobe After Effects 向けのネイティブエフェクトプラグイン集です。  
**Windows (x64)** および **macOS (Universal: Apple Silicon / Intel)** に対応しており、エフェクトパネルの **`Youmi`** カテゴリに追加されます。

---

## 収録プラグイン

| プラグイン | エフェクト表示名 | 概要 | ソースコード |
|---|---|---|---|
| **YO_Tebure** | `YO Tebure Shake` | 物理ベースの手ブレ・カメラシェイクエフェクト（プリセット選択・モーションブラー・リニアブレンド対応） | [`YO_Tebure/`](YO_Tebure/) |
| **YO_OpticBokeh** | `YO Optic Bokeh` | 光学レンズボケ・収差・ハイライト制御・フィルムグレインを細かく調整できるフル版（Rust/wgpu GPU 高速化対応） | [`YO_OpticBokeh/`](YO_OpticBokeh/) |
| **YO_OpticBokehKantan** | `YO Optic Bokeh Kantan` | `YO Optic Bokeh` の光学ボケ・収差・フィルムグレインを直感的に操作できる簡易版（GPU 高速化対応） | [`YO_OpticBokeh/`](YO_OpticBokeh/) |
| **YO_MapBilinearBlur_ja** | `YO Map Bilinear Blur` | 深度マップ（グレースケールマップ）に応じて滑らかにボケ量を制御するピラミッドブラー（日本語 UI・GPU 対応） | [`YO_MapBilinearBlur_ja/`](YO_MapBilinearBlur_ja/) |
| **YO_Shitsukan** | `YO Shitsukan` | 質感・線画保持ブラー・水彩風タッチを与えるテクスチャエフェクト（Rust/wgpu GPU 高速化対応） | [`YO_Shitsukan/`](YO_Shitsukan/) |

---

## インストール方法

ビルド済みバイナリは [Releases](../../releases) ページからダウンロードできます。

### Windows (x64)
1. `Youmi-AE-Plugins-v*-Windows-x64.zip` を解凍します。
2. 中に入っている 5 つの `.aex` ファイルを以下のフォルダにコピーします：
   ```text
   C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\
   ```
3. After Effects を再起動します。

### macOS (Universal: Apple Silicon / Intel)
1. `Youmi-AE-Plugins-v*-macOS-Universal.zip` を解凍します。
2. 中に入っている 5 つの `.plugin` バンドルを以下のフォルダにコピーします：
   ```text
   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/
   ```
3. **初回起動時のセキュリティ警告（Gatekeeper）について**  
   配布バイナリは Apple 公証（Notarization）を行っていないため、読み込み時に警告が出る場合はターミナルで以下を実行して検疫属性を解除してください：
   ```bash
   xattr -dr com.apple.quarantine "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/YO_"*.plugin
   ```
4. After Effects を再起動します。

---

## ソースからのビルド方法

### 必要な環境

- **Adobe After Effects SDK**（本リポジトリには含まれません。[Adobe Developer](https://developer.adobe.com/after-effects/) から取得し、`Examples/` フォルダのパスを指定してください。動作確認済み：AE SDK 26.5）
- **CMake 3.21 以上** および **Ninja**（または Xcode / Visual Studio）
- **C++17 コンパイラ**
  - Windows: Visual Studio 2022（`x64 Native Tools Command Prompt`）
  - macOS: Xcode Command Line Tools（`clang`, `xcrun Rez`, `lipo`）
- **Rust ツールチェーン (`cargo`)** — `YO_OpticBokeh` / `YO_OpticBokehKantan`、`YO_MapBilinearBlur_ja`（Windows GPU 版）、`YO_Shitsukan`（GPU 有効時）のビルドに必要です
  - macOS Universal ビルド時は事前に `rustup target add aarch64-apple-darwin x86_64-apple-darwin` を実行してください

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

### 2. `YO_OpticBokeh` & `YO_OpticBokehKantan`

> 1 回のビルドでフル版（`YO_OpticBokeh`）と簡易版（`YO_OpticBokehKantan`）の両方が生成されます。  
> SDK 指定には `-DAE_SDK_DIR` を使用します（GPU なし・CPU のみでビルドする場合は `-DOB_WITH_GPU=OFF`）。

```bash
# Windows (x64 Native Tools Prompt)
cmake -S YO_OpticBokeh -B build/ob -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK_DIR="C:/path/to/AfterEffectsSDK_26.5_win/Examples"
cmake --build build/ob

# macOS (Universal arm64 + x86_64)
cmake -S YO_OpticBokeh -B build/ob -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK_DIR="/path/to/AfterEffectsSDK_26.5_mac/Examples"
cmake --build build/ob
```

---

### 3. `YO_MapBilinearBlur_ja`

```powershell
# Windows (x64 Native Tools PowerShell / GPU 有効)
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo rustc --manifest-path YO_MapBilinearBlur_ja/gpu/Cargo.toml --release --locked --lib --crate-type staticlib --target-dir build/mbb_cargo
cmake -S YO_MapBilinearBlur_ja -B build/mbb -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DAE_SDK="C:/path/to/AfterEffectsSDK_26.5_win/Examples" `
  -DMBB_GPU=ON -DMBB_GPULIB="build/mbb_cargo/release/mbbgpu.lib"
cmake --build build/mbb
```

```bash
# macOS (Universal arm64 + x86_64 / CPU エンジン)
cmake -S YO_MapBilinearBlur_ja -B build/mbb -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAE_SDK="/path/to/AfterEffectsSDK_26.5_mac/Examples" -DMBB_GPU=OFF
cmake --build build/mbb
```

---

### 4. `YO_Shitsukan`

```powershell
# Windows (x64 Native Tools PowerShell / GPU 有効)
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

## ライセンス

本リポジトリのソースコードは **GNU General Public License v3.0 (GPL-3.0)** のもとで公開されています。  
詳細は [LICENSE](LICENSE) および [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) をご覧ください。
