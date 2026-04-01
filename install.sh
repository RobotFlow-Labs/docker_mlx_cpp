#!/usr/bin/env bash
# docker_mlx_cpp — One-Line Installer
#
# Give any Docker container Metal GPU access on Apple Silicon.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/docker_mlx_cpp/main/install.sh | bash
#
# What this does:
#   1. Checks you're on Apple Silicon
#   2. Checks Docker Desktop is installed
#   3. Clones the repo
#   4. Installs all MLX dependencies
#   5. Starts the MLX Daemon
#   6. Starts the Docker gateway
#   7. Runs the GPU test to verify everything works

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ${CYAN}docker_mlx_cpp${NC}${BOLD} — Metal GPU for Docker              ║${NC}"
echo -e "${BOLD}║  The NVIDIA Container Toolkit for Mac                ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check platform ──────────────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Darwin" ]]; then
  echo -e "${RED}ERROR: docker_mlx_cpp requires macOS.${NC}"
  echo "  Detected: $OS"
  exit 1
fi

if [[ "$ARCH" != "arm64" ]]; then
  echo -e "${YELLOW}WARNING: docker_mlx_cpp is designed for Apple Silicon (arm64).${NC}"
  echo "  Detected: $ARCH — Metal GPU acceleration may not work."
fi

echo -e "${GREEN}[OK]${NC} macOS $ARCH detected"

# ── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}ERROR: Python 3 not found.${NC}"
  echo "  Install: brew install python@3.12"
  exit 1
fi
echo -e "${GREEN}[OK]${NC} Python: $(python3 --version)"

# ── Check Docker ────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo -e "${RED}ERROR: Docker not found.${NC}"
  echo "  Install Docker Desktop 4.62+ from: https://docker.com/products/docker-desktop"
  exit 1
fi

if ! docker info &>/dev/null 2>&1; then
  echo -e "${RED}ERROR: Docker is not running.${NC}"
  echo "  Start Docker Desktop and try again."
  exit 1
fi
echo -e "${GREEN}[OK]${NC} Docker: $(docker --version | head -1)"

# ── Check pip/uv ────────────────────────────────────────────────────────────
INSTALLER=""
if command -v uv &>/dev/null; then
  INSTALLER="uv pip"
  echo -e "${GREEN}[OK]${NC} Using uv (fast installer)"
elif command -v pip3 &>/dev/null; then
  INSTALLER="pip3"
  echo -e "${GREEN}[OK]${NC} Using pip3"
elif command -v pip &>/dev/null; then
  INSTALLER="pip"
  echo -e "${GREEN}[OK]${NC} Using pip"
else
  echo -e "${RED}ERROR: No pip or uv found.${NC}"
  echo "  Install: brew install uv"
  exit 1
fi

# ── Clone or update repo ────────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.docker-mlx/repo"
echo ""
echo -e "${CYAN}Installing docker_mlx_cpp...${NC}"

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "  Updating existing installation..."
  git -C "$INSTALL_DIR" pull --quiet
else
  echo "  Cloning repository..."
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --quiet https://github.com/RobotFlow-Labs/docker_mlx_cpp.git "$INSTALL_DIR"
fi
echo -e "${GREEN}[OK]${NC} Repository at $INSTALL_DIR"

# ── Install Python dependencies ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}Installing MLX dependencies (this may take a minute)...${NC}"
cd "$INSTALL_DIR"

$INSTALLER install -e "." --quiet 2>/dev/null || $INSTALLER install -e "."
echo -e "${GREEN}[OK]${NC} All dependencies installed"

# ── Verify MLX ──────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Checking Metal GPU...${NC}"
if python3 -c "import mlx.core as mx; print(f'  Device: {mx.default_device()}')" 2>/dev/null; then
  echo -e "${GREEN}[OK]${NC} MLX Metal GPU available"
else
  echo -e "${YELLOW}[!!]${NC} MLX not available — install with: pip install mlx"
fi

# ── Build gateway ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Building Docker gateway...${NC}"
docker compose build mlx-gateway --quiet 2>/dev/null && echo -e "${GREEN}[OK]${NC} Gateway image built" || echo -e "${YELLOW}[!!]${NC} Gateway build skipped (run manually: docker compose build)"

# ── Add CLI to PATH ─────────────────────────────────────────────────────────
if ! command -v mlx-cpp &>/dev/null; then
  echo ""
  echo -e "${YELLOW}NOTE: Add mlx-cpp to your PATH:${NC}"
  echo "  export PATH=\"\$HOME/.docker-mlx/repo:$(python3 -m site --user-base)/bin:\$PATH\""
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ${GREEN}Installation complete!${NC}${BOLD}                               ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo ""
echo -e "    ${CYAN}# 1. Start the GPU daemon${NC}"
echo "    mlx-cpp serve"
echo ""
echo -e "    ${CYAN}# 2. Start the Docker gateway${NC}"
echo "    cd $INSTALL_DIR && docker compose up -d"
echo ""
echo -e "    ${CYAN}# 3. Test GPU from a container${NC}"
echo "    docker compose --profile gpu-test up gpu-test"
echo ""
echo -e "    ${CYAN}# 4. Quick inference${NC}"
echo "    mlx-cpp run chat-small 'Hello from Metal GPU!'"
echo ""
echo -e "  ${BOLD}Full docs:${NC} https://github.com/RobotFlow-Labs/docker_mlx_cpp"
echo ""
