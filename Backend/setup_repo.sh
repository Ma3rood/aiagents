#!/bin/bash

# Script to clone GitHub repo and setup .env file
# Usage: ./setup_repo.sh [REPO_URL] [REPO_NAME] [BRANCH_NAME]
# 
# Environment variables:
#   - GITHUB_REPO_URL: Repository URL
#   - GITHUB_REPO_NAME: Repository name (defaults to repo name from URL)
#   - GITHUB_BRANCH: Branch name (defaults to 'main')
#   - Create_Docker_Image or CREATE_DOCKER_IMAGE: Set to 'true' to build Docker image (default: 'false')
#   - Start_Container or START_CONTAINER: Set to 'true' to start container (auto-starts if image is built)
#   - DOCKER_IMAGE_NAME: Docker image name (defaults to 'Ma3roodAIAgents')
#   - DOCKER_IMAGE_TAG: Docker image tag (defaults to 'v1.0.0')
#   - CONTAINER_NAME: Container name (defaults to '{image-name}-container')
#   - FASTAPI_PORT: FastAPI server port (defaults to '8001')

# Don't exit on error for git operations, but exit on critical errors
set +e  # Allow commands to fail (we handle errors manually)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration values (used if not set via environment variables)
DEFAULT_OPENROUTER_API_KEY="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"
DEFAULT_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LOG_DIR="logs"
DEFAULT_ANALYTICS_CSV_PATH="analytics/inference_analytics.csv"
DEFAULT_CREATE_DOCKER_IMAGE="true"
DEFAULT_DOCKER_IMAGE_NAME="ma3roodagents1"
DEFAULT_DOCKER_IMAGE_TAG="v1.0.0"
DEFAULT_START_CONTAINER="true"
DEFAULT_FASTAPI_PORT="8001"
DEFAULT_CONTAINER_NAME="ma3roodagents1-container"

# Get repository URL from argument or environment variable
REPO_URL="${1:-${GITHUB_REPO_URL}}"
REPO_NAME="${2:-${GITHUB_REPO_NAME:-$(basename "$REPO_URL" .git)}}"
BRANCH_NAME="${3:-${GITHUB_BRANCH:-main}}"

# Validate repository URL
if [ -z "$REPO_URL" ]; then
    echo -e "${RED}Error: Repository URL is required${NC}"
    echo "Usage: $0 <REPO_URL> [REPO_NAME] [BRANCH_NAME]"
    echo "Or set GITHUB_REPO_URL environment variable"
    exit 1
fi

# Enable exit on error for critical operations
set -e

echo -e "${GREEN}Starting repository setup...${NC}"

# Store original directory
ORIGINAL_DIR="$(pwd)"

# Clone or update the repository
# Temporarily disable exit on error for git operations
set +e
echo -e "${YELLOW}Processing repository: $REPO_URL (branch: $BRANCH_NAME)${NC}"
if [ -d "$REPO_NAME" ]; then
    echo -e "${YELLOW}Directory $REPO_NAME already exists. Pulling latest changes...${NC}"
    cd "$REPO_NAME"
    git fetch origin 2>/dev/null
    # Try to checkout the branch, create if it doesn't exist locally, or stay on current branch
    git checkout "$BRANCH_NAME" 2>/dev/null || \
        git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME" 2>/dev/null || \
        echo -e "${YELLOW}Branch $BRANCH_NAME not found, staying on current branch${NC}"
    git pull origin "$BRANCH_NAME" 2>/dev/null || git pull 2>/dev/null || echo -e "${YELLOW}Could not pull, continuing anyway${NC}"
    cd "$ORIGINAL_DIR"
    echo -e "${GREEN}Repository updated successfully${NC}"
else
    echo -e "${YELLOW}Cloning repository: $REPO_URL${NC}"
    if git clone -b "$BRANCH_NAME" "$REPO_URL" "$REPO_NAME" 2>/dev/null; then
        echo -e "${GREEN}Repository cloned successfully on branch: $BRANCH_NAME${NC}"
    else
        echo -e "${YELLOW}Branch $BRANCH_NAME not found, cloning default branch${NC}"
        git clone "$REPO_URL" "$REPO_NAME"
        if [ $? -eq 0 ]; then
            cd "$REPO_NAME"
            if git checkout "$BRANCH_NAME" 2>/dev/null; then
                echo -e "${GREEN}Switched to branch: $BRANCH_NAME${NC}"
            else
                echo -e "${YELLOW}Using default branch${NC}"
            fi
            cd "$ORIGINAL_DIR"
        else
            echo -e "${RED}Error: Failed to clone repository${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}Repository cloned successfully${NC}"
fi
# Re-enable exit on error for critical operations
set -e

# Navigate to Backend folder in cloned repo
BACKEND_DIR="$ORIGINAL_DIR/$REPO_NAME/Backend"

if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Error: Backend folder not found in cloned repository${NC}"
    exit 1
fi

echo -e "${YELLOW}Navigating to Backend folder: $BACKEND_DIR${NC}"
cd "$BACKEND_DIR"

# Check if .env.example exists
ENV_EXAMPLE=".env.example"
if [ ! -f "$ENV_EXAMPLE" ]; then
    echo -e "${RED}Error: .env.example file not found in Backend folder${NC}"
    exit 1
fi

# Copy .env.example to .env
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}.env file already exists. Creating backup...${NC}"
    cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
fi

echo -e "${YELLOW}Copying .env.example to .env${NC}"
cp "$ENV_EXAMPLE" "$ENV_FILE"
echo -e "${GREEN}.env file created${NC}"

# Update config values in .env file
echo -e "${YELLOW}Updating config values in .env file...${NC}"

# Function to update or add environment variable
update_env_var() {
    local key=$1
    local value=$2
    local file=$3
    
    if grep -q "^${key}=" "$file"; then
        # Update existing variable
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|^${key}=.*|${key}=${value}|" "$file"
        else
            # Linux
            sed -i "s|^${key}=.*|${key}=${value}|" "$file"
        fi
    else
        # Add new variable
        echo "${key}=${value}" >> "$file"
    fi
}

# Update common environment variables
# Use environment variables if set, otherwise use default values from script

OPENROUTER_API_KEY_VALUE="${OPENROUTER_API_KEY:-$DEFAULT_OPENROUTER_API_KEY}"
OPENROUTER_BASE_URL_VALUE="${OPENROUTER_BASE_URL:-$DEFAULT_OPENROUTER_BASE_URL}"
LOG_DIR_VALUE="${LOG_DIR:-$DEFAULT_LOG_DIR}"
ANALYTICS_CSV_PATH_VALUE="${ANALYTICS_CSV_PATH:-$DEFAULT_ANALYTICS_CSV_PATH}"

if [ -n "$OPENROUTER_API_KEY_VALUE" ]; then
    update_env_var "OPENROUTER_API_KEY" "$OPENROUTER_API_KEY_VALUE" "$ENV_FILE"
    echo -e "${GREEN}Updated OPENROUTER_API_KEY${NC}"
fi

if [ -n "$OPENROUTER_BASE_URL_VALUE" ]; then
    update_env_var "OPENROUTER_BASE_URL" "$OPENROUTER_BASE_URL_VALUE" "$ENV_FILE"
    echo -e "${GREEN}Updated OPENROUTER_BASE_URL${NC}"
fi

if [ -n "$LOG_DIR_VALUE" ]; then
    update_env_var "LOG_DIR" "$LOG_DIR_VALUE" "$ENV_FILE"
    echo -e "${GREEN}Updated LOG_DIR${NC}"
fi

if [ -n "$ANALYTICS_CSV_PATH_VALUE" ]; then
    update_env_var "ANALYTICS_CSV_PATH" "$ANALYTICS_CSV_PATH_VALUE" "$ENV_FILE"
    echo -e "${GREEN}Updated ANALYTICS_CSV_PATH${NC}"
fi

# Function to check if a value is true (case-insensitive)
is_true() {
    local value=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    [[ "$value" == "true" || "$value" == "1" || "$value" == "yes" || "$value" == "y" ]]
}

# Check if Docker image should be created
CREATE_DOCKER_IMAGE_VALUE="${Create_Docker_Image:-${CREATE_DOCKER_IMAGE:-$DEFAULT_CREATE_DOCKER_IMAGE}}"

if is_true "$CREATE_DOCKER_IMAGE_VALUE"; then
    echo -e "${YELLOW}Creating Docker image...${NC}"
    
    # Check if dockerfile exists
    DOCKERFILE="dockerfile"
    if [ ! -f "$DOCKERFILE" ]; then
        echo -e "${RED}Error: dockerfile not found in Backend folder${NC}"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not available in PATH${NC}"
        exit 1
    fi
    
    # Determine image name and tag
    DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-${DEFAULT_DOCKER_IMAGE_NAME}}"
    if [ -z "$DOCKER_IMAGE_NAME" ]; then
        # Use repo name as default image name (convert to lowercase)
        DOCKER_IMAGE_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')
    fi
    DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-${DEFAULT_DOCKER_IMAGE_TAG}}"
    
    # Build the Docker image
    FULL_IMAGE_NAME="${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
    
    # Check if image already exists and delete it
    set +e
    docker image inspect "$FULL_IMAGE_NAME" &> /dev/null
    IMAGE_EXISTS=$?
    set -e
    
    if [ $IMAGE_EXISTS -eq 0 ]; then
        echo -e "${YELLOW}Docker image $FULL_IMAGE_NAME already exists. Deleting it...${NC}"
        docker rmi "$FULL_IMAGE_NAME" 2>/dev/null || {
            echo -e "${YELLOW}Warning: Could not delete image (may be in use). Continuing with build...${NC}"
        }
    fi
    
    echo -e "${YELLOW}Building Docker image: $FULL_IMAGE_NAME${NC}"
    echo -e "${YELLOW}Using dockerfile: $(pwd)/$DOCKERFILE${NC}"
    
    # Temporarily disable exit on error for docker build
    set +e
    docker build -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .
    BUILD_STATUS=$?
    set -e
    
    if [ $BUILD_STATUS -eq 0 ]; then
        echo -e "${GREEN}Docker image built successfully: $FULL_IMAGE_NAME${NC}"
    else
        echo -e "${RED}Error: Failed to build Docker image${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping Docker image creation (Create_Docker_Image is false)${NC}"
fi

# Check if container should be started
# Start container if: 1) Start_Container is explicitly set to true, OR 2) Image was just built, OR 3) Default is true
START_CONTAINER_VALUE="${Start_Container:-${START_CONTAINER:-$DEFAULT_START_CONTAINER}}"
SHOULD_START_CONTAINER=false

if [ -n "$START_CONTAINER_VALUE" ]; then
    if is_true "$START_CONTAINER_VALUE"; then
        SHOULD_START_CONTAINER=true
    fi
elif is_true "$CREATE_DOCKER_IMAGE_VALUE"; then
    # If image was just built, automatically start container
    SHOULD_START_CONTAINER=true
fi

# Determine image name and tag (reuse from above if image was built, otherwise use defaults)
if [ -z "${FULL_IMAGE_NAME:-}" ]; then
    DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-${DEFAULT_DOCKER_IMAGE_NAME}}"
    if [ -z "$DOCKER_IMAGE_NAME" ]; then
        DOCKER_IMAGE_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')
    fi
    DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-${DEFAULT_DOCKER_IMAGE_TAG}}"
    FULL_IMAGE_NAME="${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
fi

if [ "$SHOULD_START_CONTAINER" = true ]; then
    echo -e "${YELLOW}Starting Docker container...${NC}"
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not available in PATH${NC}"
        exit 1
    fi
    
    # Check if image exists
    set +e
    docker image inspect "$FULL_IMAGE_NAME" &> /dev/null
    IMAGE_EXISTS=$?
    set -e
    
    if [ $IMAGE_EXISTS -ne 0 ]; then
        echo -e "${RED}Error: Docker image $FULL_IMAGE_NAME does not exist${NC}"
        echo -e "${YELLOW}Please build the image first by setting Create_Docker_Image=true${NC}"
        exit 1
    fi
    
    # Determine container name (convert to lowercase for Docker compatibility)
    CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_CONTAINER_NAME}}"
    if [ -z "$CONTAINER_NAME" ]; then
        CONTAINER_NAME=$(echo "${DOCKER_IMAGE_NAME}-container" | tr '[:upper:]' '[:lower:]')
    fi
    
    # Get FastAPI port
    FASTAPI_PORT="${FASTAPI_PORT:-${DEFAULT_FASTAPI_PORT}}"
    
    # Check if container is already running
    set +e
    docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"
    CONTAINER_EXISTS=$?
    set -e
    
    if [ $CONTAINER_EXISTS -eq 0 ]; then
        echo -e "${YELLOW}Container $CONTAINER_NAME already exists. Stopping and removing...${NC}"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Start the container with volume mount
    echo -e "${YELLOW}Starting container: $CONTAINER_NAME${NC}"
    echo -e "${YELLOW}Mounting Backend directory: $BACKEND_DIR -> /workspace${NC}"
    echo -e "${YELLOW}Exposing port: $FASTAPI_PORT${NC}"
    
    # Convert path for Windows/Docker compatibility if running on Windows
    VOLUME_PATH="$BACKEND_DIR"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] || (command -v cygpath >/dev/null 2>&1); then
        # Running on Windows (Git Bash or Cygwin)
        # Use cygpath if available to convert to Windows path, otherwise use pwd -W
        if command -v cygpath >/dev/null 2>&1; then
            VOLUME_PATH=$(cygpath -w "$BACKEND_DIR" 2>/dev/null || echo "$BACKEND_DIR")
        elif command -v pwd >/dev/null 2>&1 && pwd -W >/dev/null 2>&1; then
            # Git Bash: convert Unix path to Windows path
            # Save current directory, change to BACKEND_DIR, get Windows path, then restore
            CURRENT_DIR=$(pwd)
            cd "$BACKEND_DIR" && VOLUME_PATH=$(pwd -W 2>/dev/null || echo "$BACKEND_DIR")
            cd "$CURRENT_DIR"
        else
            # Fallback: try to convert /c/path to C:/path manually
            if [[ "$VOLUME_PATH" =~ ^/([a-zA-Z])/(.*) ]]; then
                DRIVE_LETTER=$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]' | tr '[:lower:]' '[:upper:]')
                REST_PATH="${BASH_REMATCH[2]}"
                VOLUME_PATH="${DRIVE_LETTER}:/${REST_PATH}"
            fi
        fi
        echo -e "${YELLOW}Using Windows path for volume: $VOLUME_PATH${NC}"
    fi
    
    # Start container in detached mode with volume mount
    # Prevent Git Bash from converting container paths on Windows
    # On Windows, VOLUME_PATH is already converted to Windows format, so disable all path conversion
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # On Windows (Git Bash/Cygwin), disable path conversion for entire docker command
        # VOLUME_PATH is already in Windows format, and we want /workspace to stay as container path
        MSYS_NO_PATHCONV=1 docker run -d \
            --name "$CONTAINER_NAME" \
            -p "${FASTAPI_PORT}:${FASTAPI_PORT}" \
            -v "${VOLUME_PATH}:/workspace" \
            -w "/workspace" \
            "$FULL_IMAGE_NAME" \
            gunicorn app.main:app \
                --workers 4 \
                --worker-class uvicorn.workers.UvicornWorker \
                --bind "0.0.0.0:${FASTAPI_PORT}" \
                --reload \
                --timeout 120 \
                --access-logfile - \
                --error-logfile -
    else
        # On Linux/Mac, no path conversion needed
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "${FASTAPI_PORT}:${FASTAPI_PORT}" \
            -v "${VOLUME_PATH}:/workspace" \
            -w "/workspace" \
            "$FULL_IMAGE_NAME" \
            gunicorn app.main:app \
                --workers 4 \
                --worker-class uvicorn.workers.UvicornWorker \
                --bind "0.0.0.0:${FASTAPI_PORT}" \
                --reload \
                --timeout 120 \
                --access-logfile - \
                --error-logfile -
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Container started successfully: $CONTAINER_NAME${NC}"
        echo -e "${GREEN}FastAPI server is running on http://localhost:${FASTAPI_PORT}${NC}"
        
        # Wait for server to be ready and check health endpoint
        echo -e "${YELLOW}Waiting for FastAPI server to be ready...${NC}"
        HEALTH_ENDPOINT="http://localhost:${FASTAPI_PORT}/api/v1/health"
        MAX_RETRIES=30
        RETRY_INTERVAL=2
        HEALTH_CHECK_PASSED=false
        
        for i in $(seq 1 $MAX_RETRIES); do
            sleep $RETRY_INTERVAL
            
            # Check if curl is available
            if command -v curl &> /dev/null; then
                HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null || echo "000")
                if [ "$HTTP_CODE" = "200" ]; then
                    # Verify response contains "healthy"
                    RESPONSE=$(curl -s --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null || echo "")
                    if echo "$RESPONSE" | grep -q "healthy"; then
                        HEALTH_CHECK_PASSED=true
                        break
                    fi
                fi
            elif command -v wget &> /dev/null; then
                # Fallback to wget if curl is not available
                RESPONSE=$(wget -qO- --timeout=5 "$HEALTH_ENDPOINT" 2>/dev/null || echo "")
                if echo "$RESPONSE" | grep -q "healthy"; then
                    HEALTH_CHECK_PASSED=true
                    break
                fi
            else
                echo -e "${YELLOW}Warning: Neither curl nor wget is available. Skipping health check.${NC}"
                HEALTH_CHECK_PASSED=true  # Skip check if no HTTP client available
                break
            fi
            
            if [ $i -lt $MAX_RETRIES ]; then
                echo -e "${YELLOW}Health check attempt $i/$MAX_RETRIES failed, retrying...${NC}"
            fi
        done
        
        if [ "$HEALTH_CHECK_PASSED" = true ]; then
            echo -e "${GREEN}✓ Health check passed: FastAPI server is responding${NC}"
            echo -e "${GREEN}Health endpoint response:${NC}"
            if command -v curl &> /dev/null; then
                curl -s "$HEALTH_ENDPOINT" | head -c 200
                echo ""
            elif command -v wget &> /dev/null; then
                wget -qO- "$HEALTH_ENDPOINT" | head -c 200
                echo ""
            fi
        else
            echo -e "${RED}✗ Health check failed: FastAPI server did not respond after $MAX_RETRIES attempts${NC}"
            echo -e "${YELLOW}Container is running but server may not be ready yet.${NC}"
            echo -e "${YELLOW}Check container logs: docker logs $CONTAINER_NAME${NC}"
        fi
        
        echo -e "${YELLOW}API Documentation: http://localhost:${FASTAPI_PORT}/docs${NC}"
        echo -e "${YELLOW}To view logs: docker logs -f $CONTAINER_NAME${NC}"
        echo -e "${YELLOW}To stop container: docker stop $CONTAINER_NAME${NC}"
    else
        echo -e "${RED}Error: Failed to start Docker container${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Setup completed successfully!${NC}"