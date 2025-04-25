#// Query all network adapter which has 'jumbo frame' property and set it as 9014 bytes.
$jumboFramesList = (Get-NetAdapterAdvancedProperty -RegistryKeyword "*JumboPacket")
foreach($item in $jumboFramesList) {
  Set-NetAdapterAdvancedProperty -DisplayName $item.DisplayName -RegistryValue "9014"
}

#// Query all network adapter which has 'speed & duplex' property and set it as 1.0 gbps duplex.
$speedDuplexList = (Get-NetAdapterAdvancedProperty -RegistryKeyword "*SpeedDuplex")
foreach($item in $speedDuplexList ) {
  Set-NetAdapterAdvancedProperty -DisplayName $item.DisplayName -RegistryValue "6"
#// Registry 6 means "1.0 Gbps duplex"
}