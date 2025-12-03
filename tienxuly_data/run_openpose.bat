@echo off
set BASE_DIR=D:\test\CLIMB
set KEYPOINT_DIR=D:\test\CLIM=-09876keypoints

for /d %%F in (%BASE_DIR%\images_resized\*) do (
    echo Processing %%F ...
    cd /d D:\openpose
    bin\OpenPoseDemo.exe ^
        --image_dir "%%F" ^
        --write_json "%KEYPOINT_DIR%\%%~nF" ^
        --display 0 ^
        --render_pose 0
)
echo All done!
pause
