param(
    [Parameter(Mandatory = $true)][string]$PackageDir,
    [Parameter(Mandatory = $true)][string]$AeSdkExamples,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [switch]$WithGpu
)

$ErrorActionPreference = "Stop"

function Invoke-Exec {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = (Get-Location).Path,
        [string]$RedirectStdout = ""
    )
    Write-Host ">> $FilePath $($Arguments -join ' ')"
    if ($RedirectStdout) {
        $quotedArgs = ($Arguments | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        }) -join ' '
        $p = Start-Process -FilePath $FilePath -ArgumentList $quotedArgs -WorkingDirectory $WorkingDirectory `
            -NoNewWindow -Wait -PassThru -RedirectStandardOutput $RedirectStdout
        if ($p.ExitCode -ne 0) {
            throw "$FilePath failed with exit code $($p.ExitCode)"
        }
    } else {
        Push-Location $WorkingDirectory
        try {
            & $FilePath @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "$FilePath failed with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
    }
}

$Pkg = (Resolve-Path $PackageDir).Path
$Sdk = (Resolve-Path $AeSdkExamples).Path
$Src = Join-Path $Pkg "src"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Out = (Resolve-Path $OutDir).Path
$Obj = Join-Path $Out "obj"
New-Item -ItemType Directory -Force -Path $Obj | Out-Null

if (-not (Test-Path (Join-Path $Sdk "Headers\AE_Effect.h"))) {
    throw "AE SDK Headers/AE_Effect.h not found under $Sdk"
}

Write-Host "=== [Shitsukan] pm_bin2c ==="
Invoke-Exec "cl.exe" @(
    "/nologo", "/O2", "/EHsc", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS",
    "/Fo$Obj\", (Join-Path $Pkg "tools\pm_bin2c.cpp"), "/Fe$Out\pm_bin2c.exe"
)
Invoke-Exec (Join-Path $Out "pm_bin2c.exe") @(
    (Join-Path $Src "tables"),
    (Join-Path $Out "sk_pmath_tables.inc")
)

$gpuDefs = @()
$gpuLibs = @()
if ($WithGpu) {
    Write-Host "=== [Shitsukan] cargo skgpu ==="
    $prevRustFlags = $env:RUSTFLAGS
    $env:RUSTFLAGS = "-C target-feature=+crt-static"
    try {
        Invoke-Exec "cargo" @(
            "rustc", "--release", "--locked", "--lib", "--crate-type", "staticlib",
            "--target-dir", (Join-Path $Out "cargo")
        ) -WorkingDirectory (Join-Path $Pkg "gpu")
    } finally {
        $env:RUSTFLAGS = $prevRustFlags
    }
    $skgpuLib = Join-Path $Out "cargo\release\skgpu.lib"
    if (-not (Test-Path $skgpuLib)) { throw "skgpu.lib missing at $skgpuLib" }
    $gpuDefs = @("/DSK_GPU", "/I", (Join-Path $Pkg "gpu\include"))
    $gpuLibs = @(
        $skgpuLib, "kernel32.lib", "ntdll.lib", "userenv.lib", "ws2_32.lib",
        "dbghelp.lib", "advapi32.lib", "ole32.lib", "bcrypt.lib", "user32.lib",
        "synchronization.lib"
    )
}

$cflags = @(
    "/nologo", "/c", "/O2", "/MT", "/EHsc", "/std:c++17", "/W3", "/utf-8",
    "/DMSWindows", "/DWIN32", "/D_WINDOWS", "/DNDEBUG", "/DSK_STR_ENC=3",
    "/I", $Out
)

Write-Host "=== [Shitsukan] compile ==="
Invoke-Exec "cl.exe" ($cflags + $gpuDefs + @("/I", $Src, "/Fo$Obj\sk_core.obj", (Join-Path $Src "sk_core.cpp")))
Invoke-Exec "cl.exe" ($cflags + @("/I", $Src, "/Fo$Obj\sk_r6.obj", (Join-Path $Src "sk_r6.cpp")))
Invoke-Exec "cl.exe" ($cflags + @("/I", $Src, "/Fo$Obj\sk_strings.obj", (Join-Path $Src "sk_strings.cpp")))
Invoke-Exec "cl.exe" ($cflags + @("/I", $Src, "/Fo$Obj\sk_pmath.obj", (Join-Path $Src "sk_pmath.cpp")))
Invoke-Exec "cl.exe" ($cflags + @(
    "/I", $Src,
    "/I", (Join-Path $Sdk "Headers"),
    "/I", (Join-Path $Sdk "Headers\SP"),
    "/I", (Join-Path $Sdk "Headers\Win"),
    "/I", (Join-Path $Sdk "Util"),
    "/Fo$Obj\main.obj", (Join-Path $Src "Shitsukan.cpp")
))

Write-Host "=== [Shitsukan] PiPL ==="
Invoke-Exec "cl.exe" @(
    "/nologo", "/I", (Join-Path $Sdk "Headers"), "/I", $Src, "/EP",
    (Join-Path $Src "ShitsukanPiPL.r")
) -WorkingDirectory $Obj -RedirectStdout (Join-Path $Obj "PiPL.rr")

Invoke-Exec (Join-Path $Sdk "Resources\PiPLtool.exe") @("PiPL.rr", "PiPL.rrc") -WorkingDirectory $Obj

Invoke-Exec "cl.exe" @(
    "/nologo", "/D", "MSWindows", "/EP", "PiPL.rrc"
) -WorkingDirectory $Obj -RedirectStdout (Join-Path $Obj "PiPL.rc")

Invoke-Exec "rc.exe" @("/nologo", "/fo", "PiPL.res", "PiPL.rc") -WorkingDirectory $Obj
Invoke-Exec "rc.exe" @("/nologo", "/I", $Src, "/fo", (Join-Path $Obj "Ver.res"), (Join-Path $Src "Shitsukan.rc"))

Write-Host "=== [Shitsukan] link ==="
$outAex = Join-Path $Out "YO_Shitsukan.aex"
Invoke-Exec "link.exe" (@(
    "/nologo", "/DLL", "/MACHINE:X64", "/OPT:REF", "/OPT:ICF",
    "/OUT:$outAex", "/IMPLIB:$Obj\x.lib",
    (Join-Path $Obj "main.obj"),
    (Join-Path $Obj "sk_core.obj"),
    (Join-Path $Obj "sk_r6.obj"),
    (Join-Path $Obj "sk_strings.obj"),
    (Join-Path $Obj "sk_pmath.obj"),
    (Join-Path $Obj "PiPL.res"),
    (Join-Path $Obj "Ver.res"),
    "version.lib"
) + $gpuLibs)

Write-Host "=== OK: $outAex ==="
