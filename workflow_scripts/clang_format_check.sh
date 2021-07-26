#!/bin/bash
echo "Clang format check..."
git checkout -b workaroundBranchForCI
git cl format
DIFF=""
DIFF=`git diff`
if [ "$DIFF" == "" ]; then
  echo "Clang format check: PASS"
  exit 0
else
  echo "Clang format check: FAIL"
  git diff
  exit 1
fi