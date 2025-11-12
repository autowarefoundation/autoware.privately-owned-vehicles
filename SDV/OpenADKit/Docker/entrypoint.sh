#!/bin/bash
CLR_GREEN='\033[0;32m'
CLR_RESET='\033[0m'

echo -e "${CLR_GREEN}Visionpilot starting..${CLR_RESET}"

exec "$@"
