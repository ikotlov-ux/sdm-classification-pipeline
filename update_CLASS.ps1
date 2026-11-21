# =============================================================
# update_CLASS.ps1 - обновление скриптов в рабочей папке CLASS из GitHub
# Запуск (PowerShell):
#   powershell -ExecutionPolicy Bypass -File "I:\ikotlov\Yandex.Disk\MOS_OBL\2026\CLASS\update_CLASS.ps1"
# Что делает:
#   1. берёт свежую версию репозитория (git, если он есть, иначе zip-архив);
#   2. останавливается, если скриптов в репозитории меньше 10;
#   3. копирует все .py, requirements.txt и этот файл в CLASS (плоско);
#   4. показывает .py, которых нет в репозитории, и удаляет их только после "y";
#      удаляет папки _backup_*.
# =============================================================

$dst  = "I:\ikotlov\Yandex.Disk\MOS_OBL\2026\CLASS"
$repo = Join-Path $env:USERPROFILE "sdm-classification-pipeline"
$url  = "https://github.com/ikotlov-ux/sdm-classification-pipeline"
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1. исходники: Git, если он есть, иначе архив с GitHub
if (Get-Command git -ErrorAction SilentlyContinue) {
    if (Test-Path "$repo\.git") { git -C $repo fetch origin; git -C $repo reset --hard origin/main }
    else { if (Test-Path $repo) { Remove-Item $repo -Recurse -Force }; git clone "$url.git" $repo }
    if ($LASTEXITCODE -ne 0) { throw "git завершился с ошибкой" }
    git -C $repo log -1 --format="Версия: %h %cd %s" --date=format:"%Y-%m-%d %H:%M"
} else {
    Write-Host "git не найден - скачиваю архив" -ForegroundColor Yellow
    if (Test-Path $repo) { Remove-Item $repo -Recurse -Force }
    Invoke-WebRequest "$url/archive/refs/heads/main.zip" -OutFile "$repo.zip" -UseBasicParsing
    Expand-Archive "$repo.zip" $repo -Force; Remove-Item "$repo.zip"
}

# 2. проверка: без скриптов дальше не идём
$src = Get-ChildItem $repo -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\\.git\\' }
if ($src.Count -lt 10) { throw "Найдено только $($src.Count) скриптов в $repo - остановлено, ничего не удалено" }
Write-Host "Скриптов в репозитории: $($src.Count)"

# 3. сначала копирование
foreach ($f in $src) { Copy-Item $f.FullName (Join-Path $dst $f.Name) -Force; Write-Host "обновлён: $($f.Name)" }
Get-ChildItem $repo -Recurse -Filter requirements.txt | Select-Object -First 1 | Copy-Item -Destination $dst -Force
Get-ChildItem $repo -Recurse -Filter update_CLASS.ps1 | Select-Object -First 1 | Copy-Item -Destination $dst -Force

# 4. затем удаление лишних .py (только верхний уровень CLASS)
$names = $src.Name
$extra = Get-ChildItem $dst -File -Filter *.py | Where-Object { $names -notcontains $_.Name }
if ($extra) {
    Write-Host "`nНет в репозитории:" -ForegroundColor Yellow; $extra | ForEach-Object { Write-Host "  $($_.Name)" }
    if ((Read-Host "Удалить? (y/n)") -eq "y") { $extra | Remove-Item -Force }
}
Get-ChildItem $dst -Directory -Filter "_backup_*" | Remove-Item -Recurse -Force
