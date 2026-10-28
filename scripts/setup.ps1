$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение.' }
}
& .venv/Scripts/python.exe -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости.' }
& .venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить проект.' }
& .venv/Scripts/python.exe -m ipykernel install --prefix .venv --name garch-var --display-name 'Python (GARCH VaR .venv)'
if ($LASTEXITCODE -ne 0) { throw 'Не удалось добавить ядро Jupyter.' }
Write-Host 'Окружение готово. В VS Code выберите .venv/Scripts/python.exe для Python и ноутбука.'
