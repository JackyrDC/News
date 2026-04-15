$wsl = Get-NetFirewallHyperVVMCreator | Where-Object FriendlyName -eq "WSL"
$guid = $wsl.VMCreatorId
$guid
Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore -Name $guid
Set-NetFirewallHyperVVMSetting -Name $guid -DefaultInboundAction Allow
New-NetFirewallHyperVRule -Name "PyTorch25999" -DisplayName "Allow PyTorch 25999" -Direction Inbound -VMCreatorId $guid -Protocol TCP -LocalPorts 25999
wsl --shutdown
Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore -Name $guid
Get-NetFirewallHyperVRule -VMCreatorId $guid | Where-Object { $_.LocalPorts -eq "25999" } | Format-List Name,DisplayName,Direction,Protocol,LocalPorts,Action,Enabled