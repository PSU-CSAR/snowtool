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
    [string]$BasePythonPath,
    [string]$CertThumbprint,
    [string]$AccessLogDir
)

$ErrorActionPreference = 'Stop'

# IISAdministration has no app-pool creation cmdlet (only Get-IISAppPool);
# WebAdministration provides New-WebAppPool and the IIS: drive used below.
Import-Module IISAdministration
Import-Module WebAdministration

# Fail loudly up front rather than partway through provisioning.
if (-not (Get-WebGlobalModule -Name httpPlatformHandler -ErrorAction SilentlyContinue)) {
    throw 'The httpPlatformHandler IIS module is not installed.'
}

if (-not (Test-Path $SnowdbPath)) {
    throw "Snowdb path '$SnowdbPath' does not exist. Create it first (e.g. with 'snowtool init')."
}

# -Force does not make New-WebAppPool idempotent (an existing pool still
# throws a duplicate-collection-entry COMException); guard on existence
# instead. The pool's settings are (re)applied below either way.
if (-not (Get-IISAppPool -Name $SiteName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue)) {
    New-WebAppPool -Name $SiteName | Out-Null
}

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

# The WebAdministration writes above changed applicationHost.config on disk,
# but the shared ServerManager behind the IISAdministration cmdlets loaded
# its snapshot earlier (at first cmdlet use, the pool guard) and refuses to
# commit over a newer file ("file has changed on disk"). Do the site work
# from a fresh snapshot.
Reset-IISServerManager -Confirm:$false

$bindingInformation = "*:${Port}:${Hostname}"

# New-IISSite is no more idempotent than New-WebAppPool; drop any existing
# site and recreate so binding/path/cert changes converge on re-runs.
if (Get-IISSite -Name $SiteName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue) {
    Remove-IISSite -Name $SiteName -Confirm:$false
}

if ($Protocol -eq 'https' -and $CertThumbprint) {
    New-IISSite -Name $SiteName -PhysicalPath $PhysicalPath -BindingInformation $bindingInformation `
        -Protocol $Protocol -CertificateThumbPrint $CertThumbprint -CertStoreLocation 'Cert:\LocalMachine\My' `
        -Force | Out-Null
} elseif ($Protocol -eq 'https') {
    # New-IISSite refuses an https binding with no certificate, but the
    # binding itself is legal in IIS config -- the documented flow here is
    # binding the certificate manually in IIS Manager afterward. The raw
    # ServerManager API allows it.
    $manager = Get-IISServerManager
    $manager.Sites.Add($SiteName, $Protocol, $bindingInformation, "$PhysicalPath") | Out-Null
    $manager.CommitChanges()
} else {
    New-IISSite -Name $SiteName -PhysicalPath $PhysicalPath -BindingInformation $bindingInformation `
        -Protocol $Protocol -Force | Out-Null
}

# A freshly created site comes up on a default app pool, so always re-bind
# it to the pool configured above. WebAdministration's IIS:\Sites drive can't
# do this: its cached config view predates the site's creation within this
# process, and Set-ItemProperty NullRefs on it. Resetting gives the shared
# IISAdministration ServerManager a fresh-from-disk view guaranteed to
# include the just-committed site, whichever branch created it.
Reset-IISServerManager -Confirm:$false
$manager = Get-IISServerManager
$site = $manager.Sites[$SiteName]
$site.Applications['/'].ApplicationPoolName = $SiteName

if ($AccessLogDir) {
    $site.LogFile.Directory = "$AccessLogDir"
}

$manager.CommitChanges()

# Least privilege, granted to the pool's virtual account -- only exists once
# the pool above has been created. Read+execute to run python out of the venv
# and to read the snowdb (the API is a read-only surface -- ingest/write
# happens out-of-band via SnowDbManager, never through this process); modify
# on the site directory, which needs to write its own httpPlatform stdout log.
# The inheritable (OI)(CI) ACE on the root propagates to children on its own;
# /T (rewrite every file's ACL) would only add coverage for files with
# inheritance disabled, which none of these trees have. Propagation still
# walks the whole tree, so expect this to take minutes on a big venv/snowdb
# -- every run, deliberately: unconditional grants converge the ACLs no
# matter what state a previous run (or an admin) left them in.
icacls $VenvPath /grant "IIS AppPool\${SiteName}:(OI)(CI)RX" | Out-Null
icacls $SnowdbPath /grant "IIS AppPool\${SiteName}:(OI)(CI)RX" | Out-Null
icacls $PhysicalPath /grant "IIS AppPool\${SiteName}:(OI)(CI)M" | Out-Null

# A uv venv's python.exe is a trampoline onto the base interpreter recorded
# in pyvenv.cfg, which uv installs to the *user's* profile by default -- the
# pool account needs to read that install too, or the child process dies
# with "Access is denied" at startup.
if ($BasePythonPath) {
    icacls $BasePythonPath /grant "IIS AppPool\${SiteName}:(OI)(CI)RX" | Out-Null
}

if ($Protocol -eq 'https' -and -not $CertThumbprint) {
    Write-Warning "No certificate thumbprint given; bind the SSL certificate manually in IIS Manager (Sites > $SiteName > Edit Bindings)."
}
