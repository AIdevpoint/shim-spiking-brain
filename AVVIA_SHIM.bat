@echo off
rem SHIM: un cervello simulato a neuroni a impulsi (9 aree, 561 251 neuroni). Accende il cervello e apre la chat nel browser.
rem Prima volta: INSTALLA_SHIM.bat. La prima accensione tara i neuroni (1-2 minuti), le altre partono prima.
rem Il computer ALLA CORRENTE e nessun altro programma pesante sulla scheda grafica.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if defined SHIM_PYTHON set "PY=%SHIM_PYTHON%"
rem schede Intel: Triton compila col compilatore di Visual Studio (Build Tools) e l'SDK level-zero (SHIM_LEVEL_ZERO = la sua cartella)
set "VC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VC%" (
  call "%VC%" >nul 2>nul
  set CC=cl
)
if defined SHIM_LEVEL_ZERO set "INCLUDE=%SHIM_LEVEL_ZERO%\include;%INCLUDE%"
if defined SHIM_LEVEL_ZERO set "LIB=%SHIM_LEVEL_ZERO%\lib;%LIB%"
if not exist results mkdir results
powershell -NoProfile -Command "try { Invoke-RestMethod -TimeoutSec 3 http://127.0.0.1:8800/stato | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto pagina
start "SHIM" /min cmd /c ""%PY%" -u -m shim --porta 8800 > "%~dp0results\shim.log" 2>&1"
echo Aspetto che SHIM si svegli (la prima volta 2-4 minuti) ...
powershell -NoProfile -Command "$t=0; while ($t -lt 900) { try { $s = Invoke-RestMethod -TimeoutSec 2 http://127.0.0.1:8800/stato; Start-Sleep 3; exit 0 } catch { Start-Sleep 2; $t+=2 } }; exit 1"
if errorlevel 1 (
  echo SHIM non risponde: guarda results\shim.log
  pause
  exit /b 1
)
:pagina
start "" http://127.0.0.1:8800/
