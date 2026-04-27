@echo off
set "MAYA_LOCATION=C:\Program Files\Autodesk\Maya2026"
set "DEVKIT_LOCATION=D:\School\CGGT\CIS-6600\devkitBase"
set "MSBUILD=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
set "SLN=D:\School\CGGT\CIS-6600\AuthoringTool\MayaGaussianSplatting\GaussianSplatting\GaussianSplatting.sln"

echo Building...
echo MAYA_LOCATION=%MAYA_LOCATION%
echo DEVKIT_LOCATION=%DEVKIT_LOCATION%

"%MSBUILD%" "%SLN%" /p:Configuration=Release /p:Platform=x64 /v:m

echo.
echo ERRORLEVEL=%ERRORLEVEL%
