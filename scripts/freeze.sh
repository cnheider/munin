#!/usr/bin/env bash

out_path=""
if [ -n "$1" ]; then
 out_path=$1;
else
  out_path="frozen_env.txt";
fi;

python -m pip freeze --exclude-editable > out_path
# --user