# Tears down the IIS site + app pool for a snowtool install. Tolerant of
# either already being absent, so `snowtool iis remove` is safe to re-run.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SiteName
)

$ErrorActionPreference = 'Stop'

Import-Module IISAdministration

if (Get-IISSite -Name $SiteName -ErrorAction SilentlyContinue) {
    Remove-IISSite -Name $SiteName -Confirm:$false
}

if (Get-IISAppPool -Name $SiteName -ErrorAction SilentlyContinue) {
    Remove-IISAppPool -Name $SiteName -Confirm:$false
}
