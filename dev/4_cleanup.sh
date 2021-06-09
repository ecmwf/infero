#!/bin/bash

# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh

if [ $? == 0 ]; then
  echo "Removing ${ROOT_SRC_DIR}.."
  #rm -rf ${ROOT_SRC_DIR} \

  echo "Removing ${ROOT_BUILD_DIR}.."
  #rm -rf ${ROOT_BUILD_DIR} \
fi
