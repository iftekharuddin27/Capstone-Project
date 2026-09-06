$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-CanonicalSha256([byte[]]$Bytes) {
    $normalized = [System.Collections.Generic.List[byte]]::new()
    for ($index = 0; $index -lt $Bytes.Length; $index++) {
        if ($Bytes[$index] -eq 13) {
            $normalized.Add(10)
            if (($index + 1) -lt $Bytes.Length -and $Bytes[$index + 1] -eq 10) {
                $index++
            }
        } else {
            $normalized.Add($Bytes[$index])
        }
    }
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($normalized.ToArray()))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

$ascii = [System.Text.Encoding]::ASCII
$lfBytes = $ascii.GetBytes("text_clean,class`na,0`nb,1`n")
$crlfBytes = $ascii.GetBytes("text_clean,class`r`na,0`r`nb,1`r`n")
$changedBytes = $ascii.GetBytes("text_clean,class`na,0`nb,2`n")
Assert-True ((Get-CanonicalSha256 $lfBytes) -eq (Get-CanonicalSha256 $crlfBytes)) 'LF and CRLF canonical hashes differ'
Assert-True ((Get-CanonicalSha256 $lfBytes) -ne (Get-CanonicalSha256 $changedBytes)) 'Changed CSV value did not change canonical hash'

$configs = Get-ChildItem -LiteralPath (Join-Path $repo 'configs') -Filter 'corrected_*.json' -File
Assert-True ($configs.Count -eq 5) "Expected five corrected configuration files"
foreach ($file in $configs) {
    $config = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-True ($config.training.selection_metric -eq 'validation_macro_f1') "$($file.Name): invalid selection metric"
    Assert-True ($config.execution.allow_overwrite -eq $false) "$($file.Name): overwriting must be disabled"
    Assert-True ($config.training.random_seed -in @(13, 21, 42, 87, 101)) "$($file.Name): unsupported seed"
    foreach ($hashEntry in $config.dataset.hashes.PSObject.Properties) {
        Assert-True ($hashEntry.Value.canonical_sha256 -match '^[0-9a-f]{64}$') "$($file.Name): missing canonical_sha256 for $($hashEntry.Name)"
        if ($null -ne $hashEntry.Value.raw_sha256) {
            Assert-True ($hashEntry.Value.raw_sha256 -match '^[0-9a-f]{64}$') "$($file.Name): invalid optional raw_sha256 for $($hashEntry.Name)"
        }
    }
    if ($config.auxiliary_labels.mode -eq 'proxy') {
        Assert-True ($config.auxiliary_labels.proxy_auxiliary_labels -eq $true) "$($file.Name): proxy flag missing"
        Assert-True ($config.result_status -like 'SMOKE TEST*NOT A REPORTED RESULT') "$($file.Name): proxy output is reportable"
    }
    if ($config.model.architecture -like '*multi_task' -and $config.run_kind -eq 'full') {
        Assert-True ($config.execution.blocked -eq $true) "$($file.Name): gold multi-task run must remain blocked"
    }
}

$validator = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\data_validation.py') -Raw -Encoding UTF8
Assert-True ($validator -match 'read_bytes\(\)') 'Canonical hash must read bytes'
Assert-True ($validator -match '\.replace\(b"\\r\\n", b"\\n"\)\.replace\(b"\\r", b"\\n"\)') 'Canonical hash must normalize CRLF and CR to LF'
foreach ($control in @('missing required columns', 'expected_rows', 'unexpected labels', 'exact text duplicates')) {
    Assert-True ($validator -match [regex]::Escape($control)) "Data validation control missing: $control"
}

$training = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\training.py') -Raw -Encoding UTF8
Assert-True ($training -match '(?s)outputs\["logit_hate"\].*?batch\["labels_hate"\].*?batch\["mask_hate"\]') 'Hate loss wiring is incorrect'
Assert-True ($training -match '(?s)outputs\["logit_sarcasm"\].*?batch\["labels_sarcasm"\].*?batch\["mask_sarcasm"\]') 'Sarcasm loss wiring is incorrect'
Assert-True ($training -match 'detach\(\)\.cpu\(\)\.clone\(\)') 'Best-state CPU copy is missing'
Assert-True ($training -match 'clip_grad_norm_') 'Gradient clipping is missing'

$models = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\models.py') -Raw -Encoding UTF8
foreach ($name in @('XLMRSingleTaskModel','MultilingualDistilBERTSingleTaskModel','XLMRMultiTaskModel','MultilingualDistilBERTMultiTaskModel')) {
    Assert-True ($models -match "class $name") "Missing model class $name"
}
Assert-True (-not ($models -match '(?i)separate.encoder')) 'Single-encoder models must not be called separate encoder'

$launchers = Get-Item -LiteralPath (Join-Path $repo 'kaggle\06_corrected_smoke_test.ipynb'),(Join-Path $repo 'kaggle\07_corrected_full_reproduction.ipynb') -ErrorAction SilentlyContinue
if ($launchers.Count -eq 2) {
    foreach ($launcher in $launchers) {
        $notebook = Get-Content -LiteralPath $launcher.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        $code = (($notebook.cells | Where-Object cell_type -eq 'code' | ForEach-Object { $_.source -join '' }) -join "`n")
        Assert-True (($code -split "`n" | Where-Object { $_ -match '^\s*RUN_HEAVY\s*=\s*False\s*$' }).Count -eq 1) "$($launcher.Name): guard is not false"
        if ($launcher.Name -eq '06_corrected_smoke_test.ipynb') {
            $disabledBranch = [regex]::Match($code, '(?s)if not RUN_HEAVY:(.*?)else:').Groups[1].Value
            Assert-True (-not ($disabledBranch -match '\brepo\b')) 'Smoke notebook disabled branch references undefined repo'
            $repoAssignment = $code.IndexOf('repo = find_repo()')
            $firstRepoUse = $code.IndexOf('cwd=repo')
            Assert-True ($repoAssignment -ge 0 -and $firstRepoUse -gt $repoAssignment) 'Smoke notebook uses repo before assignment'
        }
    }
}

Write-Output "Static Stage 1A checks passed ($($configs.Count) configs)."

try {
    $version = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "python exited $LASTEXITCODE" }
    Write-Output "Python available: $version"
    & python -m compileall -q (Join-Path $repo 'corrected_pipeline') (Join-Path $repo 'tests_stage1a')
    if ($LASTEXITCODE -ne 0) { throw 'Python syntax validation failed' }
    Push-Location $repo
    try { & python -m unittest discover -s tests_stage1a -v } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw 'Python unit tests failed' }
} catch {
    Write-Output "Python checks skipped: no runnable local Python interpreter. Run compileall and unittest on Kaggle before enabling heavy execution."
}
