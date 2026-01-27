@echo off
REM Compile ICPE 2026 paper on Windows
REM Usage: compile.bat

echo ================================================
echo Compiling ICPE 2026 Paper
echo ================================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pdflatex not found!
    echo.
    echo Please install LaTeX:
    echo   - Download MiKTeX: https://miktex.org/download
    echo   - Download TeX Live: https://www.tug.org/texlive/
    echo.
    echo Or use Overleaf: https://www.overleaf.com/
    pause
    exit /b 1
)

REM Compile
echo [1/4] First pass...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [2/4] Building bibliography...
bibtex main

echo.
echo [3/4] Second pass...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [4/4] Final pass...
pdflatex -interaction=nonstopmode main.tex

REM Clean up auxiliary files
echo.
echo Cleaning up...
del /Q *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot 2>nul

echo.
echo ================================================
echo Compilation complete!
echo ================================================
echo.
echo Output: main.pdf
echo.

if exist main.pdf (
    echo SUCCESS! Paper is ready for submission.
    echo.
    for %%I in (main.pdf) do echo File size: %%~zI bytes
) else (
    echo ERROR: PDF generation failed. Check LaTeX errors above.
    pause
    exit /b 1
)

pause
