@echo off
setlocal enabledelayedexpansion

echo ========================================
echo PDF to Markdown Converter
echo ========================================
echo.

if "%~1"=="" (
    echo Usage: %0 ^<input_directory^> [output_directory]
    echo.
    echo Example: %0 "E:\PDFMark\pdf" "E:\PDFMark\output"
    echo.
    exit /b 1
)

set "INPUT_DIR=%~1"
if "%~2"=="" (
    set "OUTPUT_DIR=%~dp0output"
) else (
    set "OUTPUT_DIR=%~2"
)

if not exist "%INPUT_DIR%" (
    echo Error: Input directory does not exist: "%INPUT_DIR%"
    exit /b 1
)

echo Input directory: "%INPUT_DIR%"
echo Output directory: "%OUTPUT_DIR%"
echo.

set "CONDA_ENV=marker"

echo Activating conda environment: %CONDA_ENV%
echo.

call conda run -n %CONDA_ENV% marker "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%" --output_format markdown

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Conversion completed successfully!
    echo Output saved to: "%OUTPUT_DIR%"
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Conversion failed with error code: %errorlevel%
    echo ========================================
    exit /b %errorlevel%
)

endlocal
