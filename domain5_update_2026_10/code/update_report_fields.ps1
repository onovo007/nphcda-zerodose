param([string]$Docx, [string]$Pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$d = $w.Documents.Open($Docx)
$d.Fields.Update() | Out-Null
foreach ($t in $d.TablesOfContents) { $t.Update() | Out-Null }
foreach ($t in $d.TablesOfFigures) { $t.Update() | Out-Null }
$d.Save()
$d.ExportAsFixedFormat($Pdf, 17)
$n = $d.ComputeStatistics(2)
$d.Close()
$w.Quit()
Write-Output "pages $n"
