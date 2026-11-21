# =============================================================
# update_CLASS.ps1 - обновление скриптов в рабочей папке CLASS из GitHub
# Запуск (PowerShell):
#   powershell -ExecutionPolicy Bypass -File "I:\ikotlov\Yandex.Disk\MOS_OBL\2026\CLASS\update_CLASS.ps1"
# Что делает:
#   1. берёт свежую версию репозитория (git, если он есть, иначе zip-архив)
#      и показывает, какие коммиты пришли с прошлого обновления;
#   2. останавливается, если скриптов в репозитории меньше 10;
#   3. копирует все .py, requirements.txt и этот файл в CLASS (плоско)
#      и по каждому файлу пишет: новый / обновлён / без изменений;
#   4. показывает .py, которых нет в репозитории, и удаляет их только после "y";
#      удаляет папки _backup_*.
# Весь ход работы выводится в окно; в конце окно ждёт Enter и не закрывается
# само, в том числе при ошибке. Параметры:
#   -NoPause      не ждать Enter в конце (для запуска из другого скрипта)
#   -Dst <папка>  другая папка CLASS
# =============================================================
param(
    [string]$Dst = "I:\ikotlov\Yandex.Disk\MOS_OBL\2026\CLASS",
    [switch]$NoPause
)

$repo = Join-Path $env:USERPROFILE "sdm-classification-pipeline"
if (-not $env:USERPROFILE) { $repo = Join-Path $HOME "sdm-classification-pipeline" }
$url  = "https://github.com/ikotlov-ux/sdm-classification-pipeline"
$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
try { $Host.UI.RawUI.WindowTitle = "update_CLASS - обновление скриптов CLASS" } catch { }
# git выводит UTF-8: без этого русские описания коммитов в окне превращаются в кракозябры
$encPrev = $null
try { $encPrev = [Console]::OutputEncoding; [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

function Step($text) { Write-Host "`n$text" -ForegroundColor Cyan }

# git пишет ход работы в stderr; в Windows PowerShell при Stop это может
# считаться ошибкой, поэтому git запускается с Continue и проверяется код выхода
function Invoke-Git {
    $show = ($args | ForEach-Object { if ("$_" -match '\s') { "`"$_`"" } else { "$_" } }) -join " "
    Write-Host "> git $show" -ForegroundColor DarkGray
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & git @args
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "git $($args[0]) завершился с ошибкой (код $code)" }
}

function Get-Hash($path) {
    if (Test-Path -LiteralPath $path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash } else { $null }
}

$t0 = Get-Date
$ok = $false
$stat = [ordered]@{ "новых" = 0; "обновлено" = 0; "без изменений" = 0; "удалено" = 0 }
try {
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host " update_CLASS: обновление скриптов из GitHub   $($t0.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "Репозиторий:        $url"
    Write-Host "Папка CLASS:        $Dst"
    Write-Host "Копия репозитория:  $repo"
    Write-Host "PowerShell:         $($PSVersionTable.PSVersion)"
    if (-not (Test-Path -LiteralPath $Dst)) { throw "Папка CLASS не найдена: $Dst" }

    # 1. исходники: Git, если он есть, иначе архив с GitHub
    Step "[1/4] Получение свежей версии"
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "git: $((Get-Command git | Select-Object -First 1).Path)"
        $old = $null
        if (Test-Path (Join-Path $repo ".git")) {
            $old = (& git -C $repo rev-parse --short HEAD)
            Write-Host "Было: $old"
            Invoke-Git -C $repo fetch --progress origin
            Invoke-Git -C $repo reset --hard origin/main
        } else {
            if (Test-Path $repo) { Write-Host "Удаляю старую копию без git: $repo"; Remove-Item $repo -Recurse -Force }
            Invoke-Git clone --progress "$url.git" $repo
        }
        $new = (& git -C $repo rev-parse --short HEAD)
        Write-Host ("Версия: " + (& git --no-pager -C $repo log -1 --format="%h %cd %s" --date=format:"%Y-%m-%d %H:%M")) -ForegroundColor Green
        if ($old -and $old -ne $new) {
            Write-Host "Пришло с прошлого обновления ($old -> $new):"
            & git --no-pager -C $repo log --format="  %h %cd  %s" --date=format:"%Y-%m-%d %H:%M" "$old..$new"
        } elseif ($old) {
            Write-Host "Репозиторий не изменился с прошлого обновления."
        }
    } else {
        Write-Host "git не найден - скачиваю архив" -ForegroundColor Yellow
        if (Test-Path $repo) { Remove-Item $repo -Recurse -Force }
        Write-Host "> $url/archive/refs/heads/main.zip" -ForegroundColor DarkGray
        Invoke-WebRequest "$url/archive/refs/heads/main.zip" -OutFile "$repo.zip" -UseBasicParsing
        Write-Host ("Скачано: {0:N0} КБ" -f ((Get-Item "$repo.zip").Length / 1KB))
        Expand-Archive "$repo.zip" $repo -Force; Remove-Item "$repo.zip"
        Write-Host "Распаковано в $repo"
    }

    # 2. проверка: без скриптов дальше не идём
    Step "[2/4] Проверка"
    $src = @(Get-ChildItem $repo -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '[\\/]\.git[\\/]' })
    if ($src.Count -lt 10) { throw "Найдено только $($src.Count) скриптов в $repo - остановлено, ничего не удалено" }
    Write-Host "Скриптов в репозитории: $($src.Count)"

    # 3. сначала копирование
    Step "[3/4] Копирование в $Dst"
    $other = @()
    $other += Get-ChildItem $repo -Recurse -Filter requirements.txt | Select-Object -First 1
    $other += Get-ChildItem $repo -Recurse -Filter update_CLASS.ps1 | Select-Object -First 1
    foreach ($f in @($src + $other | Where-Object { $_ } | Sort-Object Name)) {
        $target = Join-Path $Dst $f.Name
        $before = Get-Hash $target
        if ($before -and $before -eq (Get-Hash $f.FullName)) {
            Write-Host ("  {0,-15} {1}" -f "без изменений", $f.Name) -ForegroundColor DarkGray
            $stat["без изменений"]++
            continue
        }
        Copy-Item -LiteralPath $f.FullName -Destination $target -Force
        if ($before) {
            Write-Host ("  {0,-15} {1}" -f "обновлён", $f.Name) -ForegroundColor Green
            $stat["обновлено"]++
        } else {
            Write-Host ("  {0,-15} {1}" -f "новый", $f.Name) -ForegroundColor Green
            $stat["новых"]++
        }
        if ($f.Name -eq "update_CLASS.ps1" -and $before) {
            Write-Host "                  (новая версия этого файла сработает со следующего запуска)" -ForegroundColor DarkGray
        }
    }

    # 4. затем удаление лишних .py (только верхний уровень CLASS)
    Step "[4/4] Лишние файлы"
    $names = $src.Name
    $extra = @(Get-ChildItem -LiteralPath $Dst -File -Filter *.py | Where-Object { $names -notcontains $_.Name })
    if ($extra.Count) {
        Write-Host "Нет в репозитории:" -ForegroundColor Yellow
        $extra | ForEach-Object { Write-Host "  $($_.Name)" }
        if ((Read-Host "Удалить? (y/n)") -eq "y") {
            $extra | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force; Write-Host "  удалён: $($_.Name)"; $stat["удалено"]++ }
        } else {
            Write-Host "  оставлены"
        }
    } else {
        Write-Host "Лишних .py нет"
    }
    $bak = @(Get-ChildItem -LiteralPath $Dst -Directory -Filter "_backup_*")
    foreach ($b in $bak) { Remove-Item -LiteralPath $b.FullName -Recurse -Force; Write-Host "  удалена папка: $($b.Name)" }

    $ok = $true
} catch {
    Write-Host "`nОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.InvocationInfo) { Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor DarkGray }
} finally {
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ""
    Write-Host "--------------------------------------------------------------"
    Write-Host (($stat.GetEnumerator() | ForEach-Object { "$($_.Key): $($_.Value)" }) -join "; ")
    if ($ok) { Write-Host "Готово за $sec с." -ForegroundColor Green }
    else { Write-Host "Обновление не завершено ($sec с). Файлы, скопированные до ошибки, остались." -ForegroundColor Red }
    if (-not $NoPause) { [void](Read-Host "`nНажмите Enter, чтобы закрыть окно") }
    if ($encPrev) { try { [Console]::OutputEncoding = $encPrev } catch { } }
}
