@echo off
setlocal enabledelayedexpansion

REM Script to clone GitHub repo and setup .env file
REM Usage: setup_repo.bat [REPO_URL] [REPO_NAME] [BRANCH_NAME]
REM 
REM Environment variables:
REM   - GITHUB_REPO_URL: Repository URL
REM   - GITHUB_REPO_NAME: Repository name (defaults to repo name from URL)
REM   - GITHUB_BRANCH: Branch name (defaults to 'main')
REM   - Create_Docker_Image or CREATE_DOCKER_IMAGE: Set to 'true' to build Docker image (default: 'false')
REM   - Start_Container or START_CONTAINER: Set to 'true' to start container (auto-starts if image is built)
REM   - DOCKER_IMAGE_NAME: Docker image name (defaults to 'Ma3roodAIAgents')
REM   - DOCKER_IMAGE_TAG: Docker image tag (defaults to 'v1.0.0')
REM   - CONTAINER_NAME: Container name (defaults to '{image-name}-container')
REM   - FASTAPI_PORT: FastAPI server port (defaults to '8001')

REM Default configuration values (used if not set via environment variables)
set "DEFAULT_OPENROUTER_API_KEY="
set "DEFAULT_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions"
set "DEFAULT_LOG_DIR=logs"
set "DEFAULT_ANALYTICS_CSV_PATH=analytics/inference_analytics.csv"
set "DEFAULT_CREATE_DOCKER_IMAGE=false"
set "DEFAULT_DOCKER_IMAGE_NAME=Ma3roodAIAgents"
set "DEFAULT_DOCKER_IMAGE_TAG=v1.0.0"
set "DEFAULT_START_CONTAINER=false"
set "DEFAULT_FASTAPI_PORT=8001"
set "DEFAULT_CONTAINER_NAME=ma3roodaiagents-container"

REM Get repository URL from argument or environment variable
if "%~1"=="" (
    if defined GITHUB_REPO_URL (
        set "REPO_URL=!GITHUB_REPO_URL!"
    ) else (
        echo Error: Repository URL is required
        echo Usage: %~nx0 ^<REPO_URL^> [REPO_NAME] [BRANCH_NAME]
        echo Or set GITHUB_REPO_URL environment variable
        exit /b 1
    )
) else (
    set "REPO_URL=%~1"
)

REM Get repository name
if "%~2"=="" (
    if defined GITHUB_REPO_NAME (
        set "REPO_NAME=!GITHUB_REPO_NAME!"
    ) else (
        REM Extract repo name from URL
        for %%I in ("!REPO_URL!") do set "REPO_NAME=%%~nI"
        set "REPO_NAME=!REPO_NAME:.git=!"
    )
) else (
    set "REPO_NAME=%~2"
)

REM Get branch name
if "%~3"=="" (
    if defined GITHUB_BRANCH (
        set "BRANCH_NAME=!GITHUB_BRANCH!"
    ) else (
        set "BRANCH_NAME=main"
    )
) else (
    set "BRANCH_NAME=%~3"
)

echo Starting repository setup...

REM Store original directory
set "ORIGINAL_DIR=%CD%"

REM Clone or update the repository
echo Processing repository: !REPO_URL! ^(branch: !BRANCH_NAME!^)

if exist "!REPO_NAME!" (
    echo Directory !REPO_NAME! already exists. Pulling latest changes...
    cd "!REPO_NAME!"
    git fetch origin >nul 2>&1
    git checkout !BRANCH_NAME! >nul 2>&1
    if errorlevel 1 (
        git checkout -b !BRANCH_NAME! origin/!BRANCH_NAME! >nul 2>&1
        if errorlevel 1 (
            echo Branch !BRANCH_NAME! not found, staying on current branch
        )
    )
    git pull origin !BRANCH_NAME! >nul 2>&1
    if errorlevel 1 (
        git pull >nul 2>&1
        if errorlevel 1 (
            echo Could not pull, continuing anyway
        )
    )
    cd "!ORIGINAL_DIR!"
    echo Repository updated successfully
) else (
    echo Cloning repository: !REPO_URL!
    git clone -b !BRANCH_NAME! "!REPO_URL!" "!REPO_NAME!" >nul 2>&1
    if errorlevel 1 (
        echo Branch !BRANCH_NAME! not found, cloning default branch
        git clone "!REPO_URL!" "!REPO_NAME!"
        if errorlevel 1 (
            echo Error: Failed to clone repository
            exit /b 1
        )
        cd "!REPO_NAME!"
        git checkout !BRANCH_NAME! >nul 2>&1
        if errorlevel 1 (
            echo Using default branch
        ) else (
            echo Switched to branch: !BRANCH_NAME!
        )
        cd "!ORIGINAL_DIR!"
    ) else (
        echo Repository cloned successfully on branch: !BRANCH_NAME!
    )
    echo Repository cloned successfully
)

REM Navigate to Backend folder in cloned repo
set "BACKEND_DIR=!ORIGINAL_DIR!\!REPO_NAME!\Backend"

if not exist "!BACKEND_DIR!" (
    echo Error: Backend folder not found in cloned repository
    exit /b 1
)

echo Navigating to Backend folder: !BACKEND_DIR!
cd "!BACKEND_DIR!"

REM Check if .env.example exists
set "ENV_EXAMPLE=.env.example"
if not exist "!ENV_EXAMPLE!" (
    echo Error: .env.example file not found in Backend folder
    exit /b 1
)

REM Copy .env.example to .env
set "ENV_FILE=.env"
if exist "!ENV_FILE!" (
    echo .env file already exists. Creating backup...
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DATETIME=%%I"
    set "BACKUP_FILE=!ENV_FILE!.backup.!DATETIME:~0,8!_!DATETIME:~8,6!"
    copy "!ENV_FILE!" "!BACKUP_FILE!" >nul
)

echo Copying .env.example to .env
copy "!ENV_EXAMPLE!" "!ENV_FILE!" >nul
echo .env file created

REM Update config values in .env file
echo Updating config values in .env file...

REM Get environment variable values or use defaults
if defined OPENROUTER_API_KEY (
    set "OPENROUTER_API_KEY_VALUE=!OPENROUTER_API_KEY!"
) else (
    set "OPENROUTER_API_KEY_VALUE=!DEFAULT_OPENROUTER_API_KEY!"
)

if defined OPENROUTER_BASE_URL (
    set "OPENROUTER_BASE_URL_VALUE=!OPENROUTER_BASE_URL!"
) else (
    set "OPENROUTER_BASE_URL_VALUE=!DEFAULT_OPENROUTER_BASE_URL!"
)

if defined LOG_DIR (
    set "LOG_DIR_VALUE=!LOG_DIR!"
) else (
    set "LOG_DIR_VALUE=!DEFAULT_LOG_DIR!"
)

if defined ANALYTICS_CSV_PATH (
    set "ANALYTICS_CSV_PATH_VALUE=!ANALYTICS_CSV_PATH!"
) else (
    set "ANALYTICS_CSV_PATH_VALUE=!DEFAULT_ANALYTICS_CSV_PATH!"
)

REM Update .env file variables
if not "!OPENROUTER_API_KEY_VALUE!"=="" (
    call :update_env_var "OPENROUTER_API_KEY" "!OPENROUTER_API_KEY_VALUE!" "!ENV_FILE!"
    echo Updated OPENROUTER_API_KEY
)

if not "!OPENROUTER_BASE_URL_VALUE!"=="" (
    call :update_env_var "OPENROUTER_BASE_URL" "!OPENROUTER_BASE_URL_VALUE!" "!ENV_FILE!"
    echo Updated OPENROUTER_BASE_URL
)

if not "!LOG_DIR_VALUE!"=="" (
    call :update_env_var "LOG_DIR" "!LOG_DIR_VALUE!" "!ENV_FILE!"
    echo Updated LOG_DIR
)

if not "!ANALYTICS_CSV_PATH_VALUE!"=="" (
    call :update_env_var "ANALYTICS_CSV_PATH" "!ANALYTICS_CSV_PATH_VALUE!" "!ENV_FILE!"
    echo Updated ANALYTICS_CSV_PATH
)

REM Check if Docker image should be created
if defined Create_Docker_Image (
    set "CREATE_DOCKER_IMAGE_VALUE=!Create_Docker_Image!"
) else if defined CREATE_DOCKER_IMAGE (
    set "CREATE_DOCKER_IMAGE_VALUE=!CREATE_DOCKER_IMAGE!"
) else (
    set "CREATE_DOCKER_IMAGE_VALUE=!DEFAULT_CREATE_DOCKER_IMAGE!"
)

call :is_true "!CREATE_DOCKER_IMAGE_VALUE!"
if !ERRORLEVEL! equ 0 (
    echo Creating Docker image...
    
    REM Check if dockerfile exists
    set "DOCKERFILE=dockerfile"
    if not exist "!DOCKERFILE!" (
        echo Error: dockerfile not found in Backend folder
        exit /b 1
    )
    
    REM Check if docker is available
    where docker >nul 2>&1
    if errorlevel 1 (
        echo Error: Docker is not installed or not available in PATH
        exit /b 1
    )
    
    REM Determine image name and tag
    if defined DOCKER_IMAGE_NAME (
        set "DOCKER_IMAGE_NAME=!DOCKER_IMAGE_NAME!"
    ) else (
        set "DOCKER_IMAGE_NAME=!DEFAULT_DOCKER_IMAGE_NAME!"
    )
    
    if "!DOCKER_IMAGE_NAME!"=="" (
        REM Use repo name as default image name (convert to lowercase)
        set "DOCKER_IMAGE_NAME=!REPO_NAME!"
        set "DOCKER_IMAGE_NAME=!DOCKER_IMAGE_NAME: =!"
        for /f "delims=" %%I in ('powershell -Command "[System.String]::ToLower('!DOCKER_IMAGE_NAME!')"') do set "DOCKER_IMAGE_NAME=%%I"
    )
    
    if defined DOCKER_IMAGE_TAG (
        set "DOCKER_IMAGE_TAG=!DOCKER_IMAGE_TAG!"
    ) else (
        set "DOCKER_IMAGE_TAG=!DEFAULT_DOCKER_IMAGE_TAG!"
    )
    
    REM Build the Docker image
    set "FULL_IMAGE_NAME=!DOCKER_IMAGE_NAME!:!DOCKER_IMAGE_TAG!"
    
    REM Check if image already exists and delete it
    docker image inspect "!FULL_IMAGE_NAME!" >nul 2>&1
    if not errorlevel 1 (
        echo Docker image !FULL_IMAGE_NAME! already exists. Deleting it...
        docker rmi "!FULL_IMAGE_NAME!" >nul 2>&1
        if errorlevel 1 (
            echo Warning: Could not delete image ^(may be in use^). Continuing with build...
        )
    )
    
    echo Building Docker image: !FULL_IMAGE_NAME!
    echo Using dockerfile: %CD%\!DOCKERFILE!
    
    docker build -f "!DOCKERFILE!" -t "!FULL_IMAGE_NAME!" .
    if errorlevel 1 (
        echo Error: Failed to build Docker image
        exit /b 1
    )
    echo Docker image built successfully: !FULL_IMAGE_NAME!
) else (
    echo Skipping Docker image creation ^(Create_Docker_Image is false^)
)

REM Check if container should be started
if defined Start_Container (
    set "START_CONTAINER_VALUE=!Start_Container!"
) else if defined START_CONTAINER (
    set "START_CONTAINER_VALUE=!START_CONTAINER!"
) else (
    set "START_CONTAINER_VALUE="
)

set "SHOULD_START_CONTAINER=false"

if defined START_CONTAINER_VALUE (
    call :is_true "!START_CONTAINER_VALUE!"
    if !ERRORLEVEL! equ 0 (
        set "SHOULD_START_CONTAINER=true"
    )
) else (
    call :is_true "!CREATE_DOCKER_IMAGE_VALUE!"
    if !ERRORLEVEL! equ 0 (
        REM If image was just built, automatically start container
        set "SHOULD_START_CONTAINER=true"
    )
)

REM Determine image name and tag (reuse from above if image was built, otherwise use defaults)
if not defined FULL_IMAGE_NAME (
    if defined DOCKER_IMAGE_NAME (
        set "DOCKER_IMAGE_NAME=!DOCKER_IMAGE_NAME!"
    ) else (
        set "DOCKER_IMAGE_NAME=!DEFAULT_DOCKER_IMAGE_NAME!"
    )
    
    if "!DOCKER_IMAGE_NAME!"=="" (
        set "DOCKER_IMAGE_NAME=!REPO_NAME!"
        set "DOCKER_IMAGE_NAME=!DOCKER_IMAGE_NAME: =!"
        for /f "delims=" %%I in ('powershell -Command "[System.String]::ToLower('!DOCKER_IMAGE_NAME!')"') do set "DOCKER_IMAGE_NAME=%%I"
    )
    
    if defined DOCKER_IMAGE_TAG (
        set "DOCKER_IMAGE_TAG=!DOCKER_IMAGE_TAG!"
    ) else (
        set "DOCKER_IMAGE_TAG=!DEFAULT_DOCKER_IMAGE_TAG!"
    )
    
    set "FULL_IMAGE_NAME=!DOCKER_IMAGE_NAME!:!DOCKER_IMAGE_TAG!"
)

if "!SHOULD_START_CONTAINER!"=="true" (
    echo Starting Docker container...
    
    REM Check if docker is available
    where docker >nul 2>&1
    if errorlevel 1 (
        echo Error: Docker is not installed or not available in PATH
        exit /b 1
    )
    
    REM Check if image exists
    docker image inspect "!FULL_IMAGE_NAME!" >nul 2>&1
    if errorlevel 1 (
        echo Error: Docker image !FULL_IMAGE_NAME! does not exist
        echo Please build the image first by setting Create_Docker_Image=true
        exit /b 1
    )
    
    REM Determine container name
    if defined CONTAINER_NAME (
        set "CONTAINER_NAME=!CONTAINER_NAME!"
    ) else (
        set "CONTAINER_NAME=!DEFAULT_CONTAINER_NAME!"
    )
    
    if "!CONTAINER_NAME!"=="" (
        set "CONTAINER_NAME=!DOCKER_IMAGE_NAME!-container"
        for /f "delims=" %%I in ('powershell -Command "[System.String]::ToLower('!CONTAINER_NAME!')"') do set "CONTAINER_NAME=%%I"
    )
    
    REM Get FastAPI port
    if defined FASTAPI_PORT (
        set "FASTAPI_PORT=!FASTAPI_PORT!"
    ) else (
        set "FASTAPI_PORT=!DEFAULT_FASTAPI_PORT!"
    )
    
    REM Check if container is already running
    docker ps -a --filter "name=^!CONTAINER_NAME!$" --format "{{.Names}}" | findstr /R "^!CONTAINER_NAME!$" >nul
    if not errorlevel 1 (
        echo Container !CONTAINER_NAME! already exists. Stopping and removing...
        docker stop "!CONTAINER_NAME!" >nul 2>&1
        docker rm "!CONTAINER_NAME!" >nul 2>&1
    )
    
    REM Start the container with volume mount
    echo Starting container: !CONTAINER_NAME!
    echo Mounting Backend directory: !BACKEND_DIR! -^> /workspace
    echo Exposing port: !FASTAPI_PORT!
    
    REM Start container in detached mode with volume mount
    REM Note: Docker on Windows automatically handles Windows path conversion
    docker run -d --name "!CONTAINER_NAME!" -p "!FASTAPI_PORT!:!FASTAPI_PORT!" -v "!BACKEND_DIR!:/workspace" -w /workspace "!FULL_IMAGE_NAME!" gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind "0.0.0.0:!FASTAPI_PORT!" --timeout 120 --access-logfile - --error-logfile -
    
    if not errorlevel 1 (
        echo Container started successfully: !CONTAINER_NAME!
        echo FastAPI server is running on http://localhost:!FASTAPI_PORT!
        
        REM Wait for server to be ready and check health endpoint
        echo Waiting for FastAPI server to be ready...
        set "HEALTH_ENDPOINT=http://localhost:!FASTAPI_PORT!/api/v1/health"
        set "MAX_RETRIES=30"
        set "RETRY_INTERVAL=2"
        set "HEALTH_CHECK_PASSED=false"
        
        for /L %%I in (1,1,!MAX_RETRIES!) do (
            timeout /t !RETRY_INTERVAL! /nobreak >nul
            
            REM Check if curl is available
            where curl >nul 2>&1
            if not errorlevel 1 (
                for /f "delims=" %%J in ('curl -s -o nul -w "%%{http_code}" --max-time 5 "!HEALTH_ENDPOINT!" 2^>nul') do set "HTTP_CODE=%%J"
                if "!HTTP_CODE!"=="200" (
                    for /f "delims=" %%K in ('curl -s --max-time 5 "!HEALTH_ENDPOINT!" 2^>nul') do set "RESPONSE=%%K"
                    echo !RESPONSE! | findstr /C:"healthy" >nul
                    if not errorlevel 1 (
                        set "HEALTH_CHECK_PASSED=true"
                        goto :health_check_done
                    )
                )
            ) else (
                REM Fallback to PowerShell for HTTP request
                powershell -NoProfile -Command "try { $response = Invoke-WebRequest -Uri '!HEALTH_ENDPOINT!' -TimeoutSec 5 -UseBasicParsing; if ($response.StatusCode -eq 200 -and $response.Content -match 'healthy') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
                if not errorlevel 1 (
                    set "HEALTH_CHECK_PASSED=true"
                    goto :health_check_done
                )
            )
            
            if %%I lss !MAX_RETRIES! (
                echo Health check attempt %%I/!MAX_RETRIES! failed, retrying...
            )
        )
        
        :health_check_done
        if "!HEALTH_CHECK_PASSED!"=="true" (
            echo Health check passed: FastAPI server is responding
            echo Health endpoint response:
            where curl >nul 2>&1
            if not errorlevel 1 (
                curl -s "!HEALTH_ENDPOINT!"
                echo.
            ) else (
                powershell -NoProfile -Command "try { $response = Invoke-WebRequest -Uri '!HEALTH_ENDPOINT!' -UseBasicParsing; Write-Host $response.Content } catch { }"
            )
        ) else (
            echo Health check failed: FastAPI server did not respond after !MAX_RETRIES! attempts
            echo Container is running but server may not be ready yet.
            echo Check container logs: docker logs !CONTAINER_NAME!
        )
        
        echo API Documentation: http://localhost:!FASTAPI_PORT!/docs
        echo To view logs: docker logs -f !CONTAINER_NAME!
        echo To stop container: docker stop !CONTAINER_NAME!
    ) else (
        echo Error: Failed to start Docker container
        exit /b 1
    )
)

echo Setup completed successfully!
echo .env file location: %CD%\!ENV_FILE!
echo You can now edit the .env file manually to add any additional configuration values.

endlocal
exit /b 0

REM Function to update or add environment variable in .env file
:update_env_var
setlocal enabledelayedexpansion
set "KEY=%~1"
set "VALUE=%~2"
set "FILE=%~3"

REM Check if key exists in file
findstr /B /C:"!KEY!=" "!FILE!" >nul
if not errorlevel 1 (
    REM Update existing variable using PowerShell
    powershell -Command "(Get-Content '!FILE!') -replace '^!KEY!=.*', '!KEY!=!VALUE!' | Set-Content '!FILE!'"
) else (
    REM Add new variable
    echo !KEY!=!VALUE!>>"!FILE!"
)
endlocal
exit /b 0

REM Function to check if a value is true (case-insensitive)
:is_true
setlocal enabledelayedexpansion
set "VALUE=%~1"
set "VALUE=!VALUE: =!"

REM Convert to lowercase using PowerShell
for /f "delims=" %%I in ('powershell -Command "[System.String]::ToLower('!VALUE!')"') do set "VALUE=%%I"

if "!VALUE!"=="true" exit /b 0
if "!VALUE!"=="1" exit /b 0
if "!VALUE!"=="yes" exit /b 0
if "!VALUE!"=="y" exit /b 0

exit /b 1
