$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& .venv/Scripts/python.exe -m jupyterlab outputs/garch_var_research.ipynb
