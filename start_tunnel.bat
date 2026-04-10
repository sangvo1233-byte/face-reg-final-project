@echo off
echo ============================================
echo   Phone Camera Tunnel - Diem Danh Hoc Sinh
echo ============================================
echo.
echo Dang tao tunnel toi http://localhost:8000...
echo Sau khi tunnel san sang, mo URL tren dien thoai.
echo Truy cap /phone de bat dau su dung camera dien thoai.
echo.
echo Nhan Ctrl+C de dung tunnel.
echo ============================================
echo.
cloudflared tunnel --url http://localhost:8000
