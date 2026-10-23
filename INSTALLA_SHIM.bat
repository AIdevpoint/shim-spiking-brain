@echo off
rem SHIM: installazione (una volta sola). Serve Python 3.11 (python.org, "Add to PATH") e una scheda grafica NVIDIA o Intel Arc.
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python non trovato: installa Python 3.11 da python.org & pause & exit /b 1)
if not exist venv python -m venv venv
set "PY=%~dp0venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo Scheda NVIDIA trovata: PyTorch con CUDA e Triton per Windows
  "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto errore
  "%PY%" -m pip install triton-windows || goto errore
) else (
  echo Nessuna scheda NVIDIA: PyTorch per schede Intel ^(XPU^)
  "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/xpu || goto errore
  echo ATTENZIONE: su Intel Triton compila col compilatore C++ di Visual Studio 2022 Build Tools ^(da installare a parte^)
)
"%PY%" -m pip install -r requirements.txt || goto errore
"%PY%" -c "import torch; print('scheda:', 'cuda' if torch.cuda.is_available() else ('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'NESSUNA'))"
echo Installato. Ora: AVVIA_SHIM.bat
pause
exit /b 0
:errore
echo Installazione non riuscita: guarda i messaggi sopra.
pause
exit /b 1
