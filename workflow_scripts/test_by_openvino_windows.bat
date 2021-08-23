@echo off
set TARGET=%1
echo Setup OpenVINO environment...
call "C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"
IF "%2" == "node" (
echo Run node tests by OpenVINO backend on Windows platform...
call cd %TARGET%\node
call npm run report || true
) ELSE (
echo Run WebNN End2End tests by OpenVINO backend on Windows platform...
call cd %TARGET%
call out\Release\webnn_end2end_tests.exe --gtest_output=json:..\..\%TARGET%_end2endtests.json
)