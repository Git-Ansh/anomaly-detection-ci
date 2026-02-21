#!/bin/bash
# Compile ICSME 2025 paper
# Usage: bash compile.sh

echo "================================================"
echo "Compiling ICSME 2025 Paper"
echo "================================================"

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo ""
    echo "Please install LaTeX:"
    echo "  - Windows: Install MiKTeX or TeX Live"
    echo "  - Mac: Install MacTeX"
    echo "  - Linux: sudo apt-get install texlive-full"
    echo ""
    echo "Or use Overleaf: https://www.overleaf.com/"
    exit 1
fi

# Compile
echo ""
echo "[1/4] First pass..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "[2/4] Building bibliography..."
bibtex main

echo ""
echo "[3/4] Second pass..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "[4/4] Final pass..."
pdflatex -interaction=nonstopmode main.tex

# Clean up auxiliary files
echo ""
echo "Cleaning up..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot

echo ""
echo "================================================"
echo "Compilation complete!"
echo "================================================"
echo ""
echo "Output: main.pdf"
echo ""

# Check if PDF was created
if [ -f "main.pdf" ]; then
    echo "✓ Success! Paper is ready for submission."
    echo ""
    echo "File size: $(du -h main.pdf | cut -f1)"
    echo "Page count: $(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}') pages"
else
    echo "✗ ERROR: PDF generation failed. Check LaTeX errors above."
    exit 1
fi
