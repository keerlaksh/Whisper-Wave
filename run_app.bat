@echo off
call myenv\Scripts\activate
@python.exe app.py %*
@pause
