#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${0}}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "requirements.txt not found at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR=/tmp
fi

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

case "${OS_NAME}" in
  Linux)
    MINICONDA_OS="Linux"
    case "${ARCH_NAME}" in
      x86_64|amd64)
        MINICONDA_ARCH="x86_64"
        ;;
      arm64|aarch64)
        MINICONDA_ARCH="aarch64"
        ;;
      *)
        echo "Unsupported architecture: ${ARCH_NAME}" >&2
        exit 1
        ;;
    esac
    ;;
  Darwin)
    MINICONDA_OS="MacOSX"
    case "${ARCH_NAME}" in
      x86_64|amd64)
        MINICONDA_ARCH="x86_64"
        ;;
      arm64)
        MINICONDA_ARCH="arm64"
        ;;
      *)
        echo "Unsupported architecture: ${ARCH_NAME}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported operating system: ${OS_NAME}" >&2
    exit 1
    ;;

esac

MINICONDA_INSTALLER="Miniconda3-latest-${MINICONDA_OS}-${MINICONDA_ARCH}.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
INSTALL_DIR="${HOME}/miniconda3"
INSTALLER_PATH="${TMPDIR%/}/${MINICONDA_INSTALLER}"

if [[ ! -d "${INSTALL_DIR}" ]]; then
  echo "Downloading Miniconda from ${MINICONDA_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSLo "${INSTALLER_PATH}" "${MINICONDA_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${INSTALLER_PATH}" "${MINICONDA_URL}"
  else
    echo "Neither curl nor wget is available to download Miniconda." >&2
    exit 1
  fi

  echo "Installing Miniconda to ${INSTALL_DIR}"
  bash "${INSTALLER_PATH}" -b -p "${INSTALL_DIR}"
else
  echo "Miniconda installation detected at ${INSTALL_DIR}. Skipping download."
fi

CONDA_SH="${INSTALL_DIR}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Unable to locate conda activation script at ${CONDA_SH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"

if conda env list | awk '{print $1}' | grep -qx "llm"; then
  echo "Conda environment 'llm' already exists."
else
  echo "Creating conda environment 'llm' with Python 3.11"
  conda create -y -n llm python=3.11
fi

conda activate llm

python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

echo "Environment 'llm' is ready. Activate it with: conda activate llm"
