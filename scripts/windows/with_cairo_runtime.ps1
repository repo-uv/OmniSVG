param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Command,
    [string]$CairoBin = $env:OMNISVG_CAIRO_BIN,
    [switch]$AllowMissing
)

$ErrorActionPreference = "Stop"

function Find-CairoBin {
    param([string]$Preferred)

    $candidates = @()
    if ($Preferred) {
        $candidates += $Preferred
    }

    $candidates += @(
        (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path ".cairo-runtime64\bin"),
        (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path ".cairo-runtime\bin"),
        "C:\Program Files\GTK3-Runtime Win64\bin",
        "C:\Program Files (x86)\GTK2-Runtime\bin",
        "C:\msys64\mingw64\bin",
        "C:\msys64\ucrt64\bin",
        "C:\tools\msys64\mingw64\bin",
        "C:\tools\msys64\ucrt64\bin",
        "C:\Program Files\Git\mingw64\bin"
    )

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        $dllPath = Join-Path $candidate "libcairo-2.dll"
        if (Test-Path -LiteralPath $dllPath) {
            return $candidate
        }
    }

    return $null
}

if (-not $Command -or $Command.Count -eq 0) {
    Write-Error "Usage: pwsh -File scripts/windows/with_cairo_runtime.ps1 <command> [args...]"
}

$resolvedCairoBin = Find-CairoBin -Preferred $CairoBin
if ($resolvedCairoBin) {
    $env:PATH = "$resolvedCairoBin;$env:PATH"
    Write-Host "Using Cairo runtime from $resolvedCairoBin"
} elseif (-not $AllowMissing) {
    Write-Error @"
Could not find libcairo-2.dll.

Install a Cairo runtime and retry, for example:
  - Elevated Chocolatey: choco install gtk-runtime -y
  - MSYS2: install cairo into mingw64 or ucrt64

Or set OMNISVG_CAIRO_BIN to the directory that contains libcairo-2.dll.
"@
}

& $Command[0] @($Command | Select-Object -Skip 1)
exit $LASTEXITCODE
