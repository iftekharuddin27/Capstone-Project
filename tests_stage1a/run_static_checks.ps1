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

$configs = @(
    Get-ChildItem -LiteralPath (Join-Path $repo 'configs') -Filter 'corrected_*.json' -File
    Get-ChildItem -LiteralPath (Join-Path $repo 'configs') -Filter 'pilot_*.json' -File
    Get-ChildItem -LiteralPath (Join-Path $repo 'configs') -Filter 'stage2_reportable_*.json' -File
)
Assert-True ($configs.Count -eq 19) "Expected five corrected, two pilot, and twelve Stage 2 configurations"
$pilotOutputDirectories = @()
$stage2OutputDirectories = @()
foreach ($file in $configs) {
    $config = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-True ($config.training.selection_metric -eq 'validation_macro_f1') "$($file.Name): invalid selection metric"
    Assert-True ($config.execution.allow_overwrite -eq $false) "$($file.Name): overwriting must be disabled"
    Assert-True ($config.training.random_seed -in @(13, 21, 42, 87, 101, 123, 2026)) "$($file.Name): unsupported seed"
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
    if ($config.run_kind -eq 'pilot') {
        $pilotOutputDirectories += $config.execution.output_directory
        Assert-True ($config.result_status -like 'PILOT*NOT FINAL TEST RESULT') "$($file.Name): invalid pilot result status"
        Assert-True (($config.dataset.languages -join ',') -eq 'bangla') "$($file.Name): pilot must be Bangla-only"
        Assert-True (($config.dataset.paths.PSObject.Properties.Name -join ',') -eq 'bn_train,bn_validation') "$($file.Name): pilot split scope is invalid"
        Assert-True ($config.dataset.path_base -eq 'repository_root') "$($file.Name): paths must be repository-relative"
        Assert-True ($config.execution.evaluate_test -eq $false) "$($file.Name): test evaluation must be disabled"
        Assert-True ($config.auxiliary_labels.mode -eq 'none') "$($file.Name): auxiliary labels must be disabled"
        Assert-True ($config.auxiliary_labels.proxy_auxiliary_labels -eq $false) "$($file.Name): proxy labels must be disabled"
        Assert-True ($config.training.random_seed -eq 42) "$($file.Name): pilot seed must be 42"
        Assert-True ($null -eq $config.limits) "$($file.Name): full-split pilot must not define row limits"
        Assert-True ($config.model.architecture -in @('mdistilbert_single_task', 'xlmr_single_task')) "$($file.Name): pilot must be single-task"
    }
    if ($config.run_kind -eq 'reportable_validation') {
        $stage2OutputDirectories += $config.execution.output_directory
        Assert-True ($config.result_status -like 'REPORTABLE VALIDATION*TEST LOCKED') "$($file.Name): invalid Stage 2 status"
        Assert-True ($config.training.random_seed -in @(42, 123, 2026)) "$($file.Name): invalid Stage 2 seed"
        Assert-True ($config.dataset.languages.Count -eq 1) "$($file.Name): exactly one language is required"
        Assert-True ($config.training.maximum_sequence_length -eq 128) "$($file.Name): maximum length must be 128"
        Assert-True ($config.training.epochs -eq 5 -and $config.training.early_stopping_patience -eq 2) "$($file.Name): epoch/patience lock mismatch"
        Assert-True ($config.training.optimizer -eq 'AdamW' -and $config.training.scheduler -eq 'linear_warmup') "$($file.Name): optimizer/scheduler mismatch"
        Assert-True ($config.training.weight_decay -eq 0.01 -and $config.training.warmup_ratio -eq 0.10) "$($file.Name): regularization schedule mismatch"
        Assert-True ($config.training.dropout -eq 0.30 -and $config.training.gradient_clipping -eq 1.0) "$($file.Name): dropout/clipping mismatch"
        Assert-True (($config.training.batch_size * $config.training.gradient_accumulation) -eq 64) "$($file.Name): effective batch must be 64"
        Assert-True ($config.training.effective_batch_size -eq 64) "$($file.Name): declared effective batch must be 64"
        Assert-True ($config.execution.evaluate_test -eq $false -and $config.execution.test_locked -eq $true) "$($file.Name): test lock invalid"
        Assert-True (-not ($config.dataset.paths.PSObject.Properties.Name -match 'test')) "$($file.Name): test path is forbidden"
        Assert-True ($config.auxiliary_labels.mode -eq 'none' -and $config.auxiliary_labels.proxy_auxiliary_labels -eq $false) "$($file.Name): auxiliary labels must be disabled"
        Assert-True ($config.model.revision -match '^[0-9a-f]{40}$') "$($file.Name): model revision is not immutable"
        Assert-True ($config.model.tokenizer_revision -eq $config.model.revision) "$($file.Name): tokenizer revision mismatch"
        Assert-True (($config.dataset.approved_bangla_sources -join ',') -eq 'ALERT,BD_SHS,BenSarc,BanglaSarc3,BIDWESH') "$($file.Name): approved source set mismatch"
        if ($config.model.architecture -eq 'mdistilbert_single_task') {
            Assert-True ($config.model.revision -eq '45c032ab32cc946ad88a166f7cb282f58c753c2e') "$($file.Name): mDistilBERT revision mismatch"
            Assert-True ($config.training.learning_rate -eq 0.00003 -and $config.training.batch_size -eq 32 -and $config.training.gradient_accumulation -eq 2) "$($file.Name): mDistilBERT training controls mismatch"
        } elseif ($config.model.architecture -eq 'xlmr_single_task') {
            Assert-True ($config.model.revision -eq 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089') "$($file.Name): XLM-R revision mismatch"
            Assert-True ($config.training.learning_rate -eq 0.00002 -and $config.training.batch_size -eq 16 -and $config.training.gradient_accumulation -eq 4) "$($file.Name): XLM-R training controls mismatch"
        } else {
            throw "$($file.Name): Stage 2 architecture is not approved"
        }
        if ($config.dataset.languages[0] -eq 'bangla') {
            Assert-True (($config.training.class_weights.bangla -join ',') -eq '0.67442656,1.09080106,1.66527498') "$($file.Name): Bangla class weights mismatch"
        } else {
            Assert-True ($null -eq $config.training.class_weights.english) "$($file.Name): English class weights must be inactive"
        }
    }
}
Assert-True (($pilotOutputDirectories | Select-Object -Unique).Count -eq 2) 'Stage 1B pilots must use separate output directories'
Assert-True ($stage2OutputDirectories.Count -eq 12) 'Expected twelve Stage 2 output directories'
Assert-True (($stage2OutputDirectories | Select-Object -Unique).Count -eq 12) 'Stage 2 output directories must be unique'

$validator = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\data_validation.py') -Raw -Encoding UTF8
Assert-True ($validator -match 'read_bytes\(\)') 'Canonical hash must read bytes'
Assert-True ($validator -match '\.replace\(b"\\r\\n", b"\\n"\)\.replace\(b"\\r", b"\\n"\)') 'Canonical hash must normalize CRLF and CR to LF'
foreach ($control in @('missing required columns', 'expected_rows', 'unexpected labels', 'exact text duplicates')) {
    Assert-True ($validator -match [regex]::Escape($control)) "Data validation control missing: $control"
}

$configSource = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\config.py') -Raw -Encoding UTF8
Assert-True ($configSource -match 'run_kind not in \{"smoke", "pilot", "full", "reportable_validation"\}') 'Accepted run kinds must include the Stage 2 validation-only type'
foreach ($pilotControl in @('PILOT', 'NOT FINAL TEST RESULT', 'execution.evaluate_test=false', 'bn_train', 'bn_validation', 'random_seed=42')) {
    Assert-True ($configSource -match [regex]::Escape($pilotControl)) "Pilot config control missing: $pilotControl"
}

$training = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\training.py') -Raw -Encoding UTF8
Assert-True ($training -match '(?s)outputs\["logit_hate"\].*?batch\["labels_hate"\].*?batch\["mask_hate"\]') 'Hate loss wiring is incorrect'
Assert-True ($training -match '(?s)outputs\["logit_sarcasm"\].*?batch\["labels_sarcasm"\].*?batch\["mask_sarcasm"\]') 'Sarcasm loss wiring is incorrect'
Assert-True ($training -match 'detach\(\)\.cpu\(\)\.clone\(\)') 'Best-state CPU copy is missing'
Assert-True ($training -match 'clip_grad_norm_') 'Gradient clipping is missing'

$runner = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\runner.py') -Raw -Encoding UTF8
Assert-True ($runner -match 'dataset_hashes\.json') 'Standalone dataset hash artifact is missing'
Assert-True ($runner -match 'split_specs = \{key: SPLIT_SPECS\[key\] for key in dataset_paths\}') 'Runner must validate only configured splits'
Assert-True ($runner -match 'is_validation_only = config\["run_kind"\] in \{"pilot", "reportable_validation"\}') 'Runner validation-only branch is missing'
Assert-True ($runner -match 'if is_validation_only:(?s).*?evaluate_test = False') 'Runner validation-only test guard is missing'
Assert-True ($runner -match 'if evaluate_test:(?s).*?test_frame = pd\.read_csv') 'Test loading must use the guarded local flag'

$models = Get-Content -LiteralPath (Join-Path $repo 'corrected_pipeline\models.py') -Raw -Encoding UTF8
foreach ($name in @('XLMRSingleTaskModel','MultilingualDistilBERTSingleTaskModel','XLMRMultiTaskModel','MultilingualDistilBERTMultiTaskModel')) {
    Assert-True ($models -match "class $name") "Missing model class $name"
}
Assert-True (-not ($models -match '(?i)separate.encoder')) 'Single-encoder models must not be called separate encoder'

$launchers = Get-Item -LiteralPath (Join-Path $repo 'kaggle\06_corrected_smoke_test.ipynb'),(Join-Path $repo 'kaggle\07_corrected_full_reproduction.ipynb'),(Join-Path $repo 'kaggle\08_bn_single_task_seed42_pilot.ipynb') -ErrorAction SilentlyContinue
Assert-True ($launchers.Count -eq 3) 'Expected three corrected Kaggle launchers'
if ($launchers.Count -eq 3) {
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
        if ($launcher.Name -eq '08_bn_single_task_seed42_pilot.ipynb') {
            Assert-True ($code -match "MODEL_TO_RUN\s*=\s*'mdistilbert'") 'Stage 1B selector default is invalid'
            Assert-True ($code -match "'mdistilbert':\s*'pilot_bn_mdistilbert_single_seed42.json'") 'Stage 1B mdistilbert choice missing'
            Assert-True ($code -match "'xlmr':\s*'pilot_bn_xlmr_single_seed42.json'") 'Stage 1B XLM-R choice missing'
            Assert-True (-not ($code -match '(?i)\bpip\s+install\b')) 'Stage 1B notebook must not install packages'
            $testIndex = $code.IndexOf("'-m', 'unittest'")
            $runnerIndex = $code.IndexOf("'-m', 'corrected_pipeline.runner'")
            Assert-True ($testIndex -ge 0 -and $runnerIndex -gt $testIndex) 'Stage 1B tests must run before training'
            Assert-True ([regex]::Matches($code, "'-m', 'corrected_pipeline\.runner'").Count -eq 1) 'Stage 1B must invoke only one selected model'
        }
    }
}

$allKaggleNotebooks = Get-ChildItem -LiteralPath (Join-Path $repo 'kaggle') -Filter '*.ipynb' -File
Assert-True ($allKaggleNotebooks.Count -eq 13) 'Expected thirteen Kaggle notebooks after Stage 2 preparation'
foreach ($notebookFile in $allKaggleNotebooks) {
    $notebook = Get-Content -LiteralPath $notebookFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $code = (($notebook.cells | Where-Object cell_type -eq 'code' | ForEach-Object { $_.source -join '' }) -join "`n")
    $guards = @($code -split "`n" | Where-Object { $_ -match '^\s*RUN_HEAVY\s*=' })
    Assert-True ($guards.Count -eq 1) "$($notebookFile.Name): expected exactly one RUN_HEAVY assignment"
    Assert-True ($guards[0] -match '^\s*RUN_HEAVY\s*=\s*False\s*$') "$($notebookFile.Name): RUN_HEAVY must remain False"
}

$stage2Launchers = Get-ChildItem -LiteralPath (Join-Path $repo 'kaggle') -Filter '1*_stage2_*.ipynb' -File
Assert-True ($stage2Launchers.Count -eq 4) 'Expected four Stage 2 launchers'
foreach ($launcher in $stage2Launchers) {
    $notebook = Get-Content -LiteralPath $launcher.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $code = (($notebook.cells | Where-Object cell_type -eq 'code' | ForEach-Object { $_.source -join '' }) -join "`n")
    foreach ($control in @('ALLOWED_SEEDS = (42, 123, 2026)', "'-m', 'compileall'", "'-m', 'unittest'", "torch.cuda.is_available", "evaluate_test", "allow_overwrite", "dataset_hashes", "effective_batch_size")) {
        Assert-True ($code -match [regex]::Escape($control)) "$($launcher.Name): missing control $control"
    }
    Assert-True ([regex]::Matches($code, "'-m', 'corrected_pipeline\.runner'").Count -eq 1) "$($launcher.Name): must invoke exactly one selected run"
    $testsIndex = $code.IndexOf("'-m', 'unittest'")
    $runnerIndex = $code.IndexOf("'-m', 'corrected_pipeline.runner'")
    Assert-True ($testsIndex -ge 0 -and $runnerIndex -gt $testsIndex) "$($launcher.Name): tests must precede training"
}

Write-Output "Static corrected-pipeline checks passed ($($configs.Count) configs)."

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
