# Tears down the IIS site + app pool for a snowtool install, plus the
# app-pool account's permission grants left by install_site.ps1. Tolerant of
# everything already being absent, so `snowtool iis remove` is safe to re-run.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SiteName,
    [string]$VenvPath,
    [string]$BasePythonPath,
    [string]$SnowdbPath,
    [string]$PhysicalPath
)

$ErrorActionPreference = 'Stop'

# IISAdministration has no app-pool removal cmdlet (only Get-IISAppPool);
# WebAdministration provides Remove-WebAppPool.
Import-Module IISAdministration
Import-Module WebAdministration

if (Get-IISSite -Name $SiteName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue) {
    Remove-IISSite -Name $SiteName -Confirm:$false
}

if (Get-IISAppPool -Name $SiteName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue) {
    Remove-WebAppPool -Name $SiteName
}

# The install grants name the pool's virtual account, whose SID derives from
# the pool name alone -- they would silently re-attach to any future pool
# recreated under the same name, so remove them wherever install granted
# them. Resolving the account name does not require the pool to still exist,
# and like the grants themselves, removal propagates through the whole tree
# -- slow on a big venv/snowdb.
foreach ($grantedPath in @($VenvPath, $BasePythonPath, $SnowdbPath, $PhysicalPath)) {
    if ($grantedPath -and (Test-Path $grantedPath)) {
        icacls $grantedPath /remove:g "IIS AppPool\$SiteName" | Out-Null
    }
}
