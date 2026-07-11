# Idempotently provisions (or updates) the IIS app pool + site for a snowtool
# install. Invoked by snowtool.cli._iis.provisioning.install_args via
# `powershell -File install_site.ps1 -SiteName ... -PhysicalPath ...` -- every
# value below arrives as a bound script parameter, never interpolated into a
# -Command string, so a site name/hostname can't inject PowerShell.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SiteName,
    [Parameter(Mandatory)][string]$PhysicalPath,
    [Parameter(Mandatory)][string]$VenvPath,
    [Parameter(Mandatory)][string]$SnowdbPath,
    [Parameter(Mandatory)][string]$Hostname,
    [Parameter(Mandatory)][int]$Port,
    [Parameter(Mandatory)][ValidateSet('http', 'https')][string]$Protocol,
    [Parameter(Mandatory)][string]$RecycleTime,
    [string]$CertThumbprint,
    [string]$AccessLogDir
)

$ErrorActionPreference = 'Stop'

Import-Module IISAdministration

# Fail loudly up front rather than partway through provisioning.
if (-not (Get-WebGlobalModule -Name httpPlatformHandler -ErrorAction SilentlyContinue)) {
    throw 'The httpPlatformHandler IIS module is not installed.'
}

if (-not (Test-Path $SnowdbPath)) {
    throw "Snowdb path '$SnowdbPath' does not exist. Create it first (e.g. with 'snowtool init')."
}

New-IISAppPool -Name $SiteName -Force | Out-Null

# AlwaysRunning + a zero idle timeout keep the app-pool worker (and its
# httpPlatformHandler child process) up even with no traffic -- without this
# the pool idles out (default 20 min) and silently kills the API until the
# next request restarts it. The periodic recycle is pinned to a fixed quiet
# hour instead of IIS's ~29h default, so restarts are predictable rather than
# landing mid-traffic.
Set-ItemProperty "IIS:\AppPools\$SiteName" -Name managedRuntimeVersion -Value ''
Set-ItemProperty "IIS:\AppPools\$SiteName" -Name startMode -Value 'AlwaysRunning'
Set-ItemProperty "IIS:\AppPools\$SiteName" -Name processModel.idleTimeout -Value '00:00:00'
Set-ItemProperty "IIS:\AppPools\$SiteName" -Name recycling.periodicRestart.time -Value $RecycleTime

$bindingInformation = "*:${Port}:${Hostname}"

if ($Protocol -eq 'https' -and $CertThumbprint) {
    New-IISSite -Name $SiteName -PhysicalPath $PhysicalPath -BindingInformation $bindingInformation `
        -Protocol $Protocol -CertificateThumbPrint $CertThumbprint -CertStoreLocation 'Cert:\LocalMachine\My' `
        -Force | Out-Null
} else {
    New-IISSite -Name $SiteName -PhysicalPath $PhysicalPath -BindingInformation $bindingInformation `
        -Protocol $Protocol -Force | Out-Null
}

# New-IISSite -Force may leave the site on a pool of its own name from a prior
# run; always re-bind it to the pool we just configured.
Set-ItemProperty "IIS:\Sites\$SiteName" -Name applicationPool -Value $SiteName

if ($AccessLogDir) {
    Set-ItemProperty "IIS:\Sites\$SiteName" -Name logFile.directory -Value $AccessLogDir
}

# Least privilege, granted to the pool's virtual account -- only exists once
# the pool above has been created. Read+execute to run python out of the venv
# and to read the snowdb (the API is a read-only surface -- ingest/write
# happens out-of-band via SnowDbManager, never through this process); modify
# on the site directory, which needs to write its own httpPlatform stdout log.
icacls $VenvPath /grant "IIS AppPool\${SiteName}:(OI)(CI)RX" /T | Out-Null
icacls $SnowdbPath /grant "IIS AppPool\${SiteName}:(OI)(CI)RX" /T | Out-Null
icacls $PhysicalPath /grant "IIS AppPool\${SiteName}:(OI)(CI)M" /T | Out-Null

if ($Protocol -eq 'https' -and -not $CertThumbprint) {
    Write-Warning "No certificate thumbprint given; bind the SSL certificate manually in IIS Manager (Sites > $SiteName > Edit Bindings)."
}
