$jumboFramesList = (Get-NetAdapterAdvancedProperty -RegistryKeyword "*JumboPacket")
foreach($item in $jumboFramesList) {
  Reset-NetAdapterAdvancedProperty -Name * -DisplayName $item.DisplayName
}

$speedDuplexList = (Get-NetAdapterAdvancedProperty -RegistryKeyword "*SpeedDuplex")
foreach($item in $speedDuplexList) {
  Reset-NetAdapterAdvancedProperty -Name * -DisplayName $item.DisplayName
}