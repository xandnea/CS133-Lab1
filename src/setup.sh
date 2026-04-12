#!/bin/bash
if [ "$(uname -s)" == 'Darwin' ]; then
  xcode-select --install
  if ! which brew > /dev/null 2>&1; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  # Ensure brew is in PATH
  if [ -x "/opt/homebrew/bin/brew" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
  elif [ -x "/usr/local/bin/brew" ]; then
    export PATH="/usr/local/bin:$PATH"
  fi
  brew install llvm zip
else
  sudo apt update
  sudo env DEBIAN_FRONTEND=noninteractive apt install build-essential zip -y
fi
