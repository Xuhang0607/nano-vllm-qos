param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# nano-vLLM QoS one-click launcher for Windows + WSL2.
# Edit the variables below if your WSL distro, model path, or port changes.

# Leave this empty to use the default WSL distro. This is the most robust option
# when double-clicking from Windows.
$WslDistro = ""

$ProjectDir = "/mnt/d/nano-vllm-qos"
$ModelDir = "/mnt/d/models/Qwen3-0.6B"
$VenvDir = "/home/xuhang/.venvs/nanovllm-qos"
$Port = "8020"

# Core serving features.
$EnableMooncake = "1"
$SchedulingPolicy = "pals"
$PrefixCacheBackend = "radix"
$MaxModelLen = "40960"
$MaxNumSeqs = "64"
$GpuMemoryUtilization = "0.90"

# KV reclaim: lossless partial KV reclaim under memory pressure.
$KvReclaimPolicy = "slo_aware"
$KvReclaimMinKeepRatio = "0.0"
$KvReclaimMaxKeepRatio = "0.75"
$KvReclaimBudgetScaleMs = "1000"
$KvReclaimTargetFreeBlocks = "2"

# KV compression: use "none" for daily chat, or "query_aware" for the compression demo.
$KvCompressionPolicy = "none"
$KvCompressionSinkBlocks = "1"
$KvCompressionRecentBlocks = "8"
$KvCompressionImportanceBlocks = "2"
$KvCompressionQueryTokens = "64"
$KvCompressionTriggerFreeRatio = "0.15"

# Optional: set this to "18" to force a smaller KV cache for pressure/compression demos.
$NumKvCacheBlocks = ""

function Quote-BashValue {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$envPairs = [ordered]@{
    PORT                              = $Port
    PROJECT_DIR                       = $ProjectDir
    MODEL_DIR                         = $ModelDir
    VENV_DIR                          = $VenvDir
    ENABLE_MOONCAKE                   = $EnableMooncake
    SCHEDULING_POLICY                 = $SchedulingPolicy
    PREFIX_CACHE_BACKEND              = $PrefixCacheBackend
    MAX_MODEL_LEN                     = $MaxModelLen
    MAX_NUM_SEQS                      = $MaxNumSeqs
    GPU_MEMORY_UTILIZATION            = $GpuMemoryUtilization
    KV_RECLAIM_POLICY                 = $KvReclaimPolicy
    KV_RECLAIM_MIN_KEEP_RATIO         = $KvReclaimMinKeepRatio
    KV_RECLAIM_MAX_KEEP_RATIO         = $KvReclaimMaxKeepRatio
    KV_RECLAIM_BUDGET_SCALE_MS        = $KvReclaimBudgetScaleMs
    KV_RECLAIM_TARGET_FREE_BLOCKS     = $KvReclaimTargetFreeBlocks
    KV_COMPRESSION_POLICY             = $KvCompressionPolicy
    KV_COMPRESSION_SINK_BLOCKS        = $KvCompressionSinkBlocks
    KV_COMPRESSION_RECENT_BLOCKS      = $KvCompressionRecentBlocks
    KV_COMPRESSION_IMPORTANCE_BLOCKS  = $KvCompressionImportanceBlocks
    KV_COMPRESSION_QUERY_TOKENS       = $KvCompressionQueryTokens
    KV_COMPRESSION_TRIGGER_FREE_RATIO = $KvCompressionTriggerFreeRatio
}

if ($NumKvCacheBlocks) {
    $envPairs["NUM_KVCACHE_BLOCKS"] = $NumKvCacheBlocks
}

$exports = foreach ($item in $envPairs.GetEnumerator()) {
    "$($item.Key)=$(Quote-BashValue $item.Value)"
}

$bashCommand = "cd $(Quote-BashValue $ProjectDir) && source $(Quote-BashValue "$VenvDir/bin/activate") && " +
    ($exports -join " ") +
    " bash scripts/start_full_service_wsl.sh"

$wslArgs = @()
if ($WslDistro) {
    $wslArgs += @("-d", $WslDistro)
}
$wslArgs += @("--", "bash", "-lc", $bashCommand)

Write-Host ""
Write-Host "[nano-vLLM QoS] Starting service in WSL..."
Write-Host "  WSL distro : " -NoNewline
if ($WslDistro) {
    Write-Host $WslDistro
} else {
    Write-Host "default"
}
Write-Host "  Project    : $ProjectDir"
Write-Host "  Model      : $ModelDir"
Write-Host "  Web UI     : http://127.0.0.1:$Port/"
Write-Host ""
Write-Host "Keep this window open while using the service."
Write-Host "Press Ctrl+C in this window to stop nano-vLLM."
Write-Host ""

if ($DryRun) {
    Write-Host "[nano-vLLM QoS] Dry run only. WSL arguments:"
    Write-Host ($wslArgs -join " ")
    exit 0
}

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile",
    "-Command",
    "Start-Sleep -Seconds 15; Start-Process 'http://127.0.0.1:$Port/'"
)

& wsl.exe @wslArgs
