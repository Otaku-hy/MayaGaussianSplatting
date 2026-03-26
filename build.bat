@echo off
set MAYA_LOCATION=C:\Program Files\Autodesk\Maya2026
set DEVKIT_LOCATION=D:\School\CGGT\CIS-6600\devkitBase
set MSBUILD=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe
set SLN=D:\School\CGGT\CIS-6600\AuthoringTool\MayaGaussianSplatting\GaussianSplatting\GaussianSplatting.sln

echo Building GaussianSplatting plugin...
echo MAYA_LOCATION=%MAYA_LOCATION%
echo DEVKIT_LOCATION=%DEVKIT_LOCATION%

"%MSBUILD%" "%SLN%" /p:Configuration=Release /p:Platform=x64 /p:MAYA_LOCATION="%MAYA_LOCATION%" /p:DEVKIT_LOCATION="%DEVKIT_LOCATION%" /v:m

if %ERRORLEVEL%==0 (
    echo.
    echo Build SUCCESS
    echo Plugin: D:\School\CGGT\CIS-6600\devkitBase\devkit\plug-ins\GaussianSplatting.mll
) else (
    echo.
    echo Build FAILED - check errors above
)
