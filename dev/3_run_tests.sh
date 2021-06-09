#!/bin/bash


# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh


echo "cd in ${INFERO_BUILD_DIR}.."
cd ${INFERO_BUILD_DIR} && \

# run infero tests
make test
