<#
.SYNOPSIS
    Sets up the project environment and runs the full energy allocation pipeline.
.DESCRIPTION
    Finds or installs Python 3.12, creates .venv if needed, installs requirements,
    runs the constraint walkthrough and all documented scenarios, saves solver
    summaries inside .venv, then launches the interactive local dashboard.
.EXAMPLE
    .\run_pipeline.ps1
.EXAMPLE
    .\run_pipeline.ps1 -Shots 512 -MaxIterations 50
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 100000)]
    [int]$Shots = 1024,

    [ValidateRange(1, 10000)]
    [int]$MaxIterations = 100,

    [ValidateRange(1, 65535)]
    [int]$DashboardPort = 8765
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Find-Python312 {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            & $launcher.Source -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Executable = $launcher.Source; Prefix = @("-3.12") }
            }
        }
        catch {
            # An unavailable Windows app-execution alias should not block path checks.
        }
    }

    $candidates = @()
    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE "AppData\Local\Programs\Python\Python312\python.exe"
    }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    }
    $candidates += "C:\Program Files\Python312\python.exe"
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            try {
                & $candidate -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return @{ Executable = $candidate; Prefix = @() }
                }
            }
            catch {
                # Continue in case another standard Python installation is present.
            }
        }
    }
    return $null
}

Push-Location $ProjectRoot
try {
    Write-Host "[1/6] Checking for Python 3.12..." -ForegroundColor Cyan
    $python = Find-Python312
    if (-not $python) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "Python 3.12 was not found and winget is unavailable. Install Python 3.12, then rerun this script."
        }
        Write-Host "Python 3.12 is missing; installing it with winget..." -ForegroundColor Yellow
        Invoke-Checked $winget.Source @(
            "install", "--id", "Python.Python.3.12", "--exact",
            "--accept-package-agreements", "--accept-source-agreements"
        )
        $python = Find-Python312
        if (-not $python) {
            throw "winget completed, but Python 3.12 could not be located. Open a new terminal and rerun this script."
        }
    }
    & $python.Executable @($python.Prefix) --version
    if ($LASTEXITCODE -ne 0) { throw "Could not run Python 3.12." }

    Write-Host "[2/6] Creating the virtual environment if needed..." -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Invoke-Checked $python.Executable (@($python.Prefix) + @("-m", "venv", ".venv"))
    }

    $LocalTemp = Join-Path $ProjectRoot ".venv\temp"
    $MatplotlibCache = Join-Path $ProjectRoot ".venv\matplotlib-cache"
    New-Item -ItemType Directory -Force -Path $LocalTemp, $MatplotlibCache | Out-Null
    $env:TEMP = $LocalTemp
    $env:TMP = $LocalTemp
    $env:MPLCONFIGDIR = $MatplotlibCache

    Write-Host "[3/6] Installing project dependencies..." -ForegroundColor Cyan
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked $VenvPython @("-m", "pip", "check")

    Write-Host "[4/6] Running the constraint walkthrough..." -ForegroundColor Cyan
    Invoke-Checked $VenvPython @("src\show_steps.py")

    Write-Host "[5/6] Running the documented scenarios..." -ForegroundColor Cyan
    $scenarios = @("base", "renewable_shortfall", "high_demand")
    $ScenarioResults = Join-Path $ProjectRoot ".venv\pipeline-results"
    New-Item -ItemType Directory -Force -Path $ScenarioResults | Out-Null
    foreach ($scenario in $scenarios) {
        Invoke-Checked $VenvPython @(
            "main.py", "--scenario", $scenario,
            "--shots", "$Shots", "--maxiter", "$MaxIterations",
            "--json-output", (Join-Path $ScenarioResults "$scenario.json")
        )
    }

    Write-Host "[6/6] Launching the interactive local dashboard..." -ForegroundColor Cyan
    Invoke-Checked $VenvPython @(
        "-m", "src.visualize", "--port", "$DashboardPort",
        "--results-dir", $ScenarioResults
    )
}
finally {
    Pop-Location
}
