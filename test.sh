#!/usr/bin/env bash

ENVNAME=$(basename `pwd`)

echo "(1) REMOVING CONDA ENV \"${ENVNAME}\""
conda env remove -yq -n $ENVNAME &> /dev/null
echo "==================================================="
echo "(2) CREATING NEW CONDA ENV \"${ENVNAME}\""
conda create -yq -n $ENVNAME --file conda.txt 1> /dev/null
echo "==================================================="
echo "(3) ACTIVATING CONDA ENV \"${ENVNAME}\""
(source activate $ENVNAME
echo "==================================================="
echo "(4) EDITING GLOBAL VARIABLES"
# http://conda.pydata.org/docs/troubleshooting.html
echo "    Unsetting PYTHONPATH"
unset PYTHONPATH
echo "    Unsetting PYTHONHOME"
unset PYTHONHOME
echo "    Editing PATH"
PATH=$(conda info | awk '/default environment : / {print $4}')/bin:/bin:/usr/local/bin:/usr/bin
echo "==================================================="
echo "(5) INSTALLING \"$(basename `pwd`)\""
export PIP_CONFIG_FILE=$(pwd)/.pip/pip.conf
pip install -e .
py.test -f)