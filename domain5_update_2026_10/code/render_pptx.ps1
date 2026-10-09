param([string]$Pptx, [string]$OutDir, [string]$Pdf = "")
New-Item -ItemType Directory -Force $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($Pptx, $true, $false, $false)
$i = 1
foreach ($s in $pres.Slides) {
    $s.Export((Join-Path $OutDir ("slide_{0:D2}.png" -f $i)), "PNG", 1600, 900)
    $i++
}
if ($Pdf -ne "") { $pres.SaveAs($Pdf, 32) }
$pres.Close()
$app.Quit()
Write-Output "rendered $($i - 1)"
