#!/usr/bin/env bash

ENVNAME="$(basename `pwd`)"

echo "(1) REMOVING CONDA ENV \"${ENVNAME}\""
conda env remove -yq -n ${ENVNAME} &> /dev/null
echo "==================================================="
echo "(2) CREATING NEW CONDA ENV \"${ENVNAME}\""
conda create -yq -n ${ENVNAME} -c conda-forge --file conda.txt
echo "==================================================="
echo "(3) ACTIVATING CONDA ENV \"${ENVNAME}\""
(PYTHON_CURRENT=$(which python | xargs dirname)
source activate ${ENVNAME}
echo "==================================================="
echo "(4) EDITING GLOBAL VARIABLES"
# http://conda.pydata.org/docs/troubleshooting.html
echo "    Unsetting PYTHONPATH"
unset PYTHONPATH
echo "    Unsetting PYTHONHOME"
unset PYTHONHOME
echo "    Editing PATH"
PATH=$(echo "${PATH}" | sed "s@${PYTHON_CURRENT}@@g")
echo "==================================================="
echo "(5) INSTALLING \"${ENVNAME}\""
pip install -e .
py.test -f)