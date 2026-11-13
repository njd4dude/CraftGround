@echo off
rem Run a Python script with the C++ compiler and the exp_craftground conda env set up.
rem Usage:  run.bat train.py --timesteps 200000
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
call C:\Users\njdon\miniconda3\Scripts\activate.bat exp_craftground
cd /d "%~dp0"
python %*
