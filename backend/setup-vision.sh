#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -d /Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk ]; then
  export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk
fi
mkdir -p bin
clang -O2 -fobjc-arc -framework Foundation -framework Vision vision-ocr.m -o bin/vision-ocr
echo "Local Apple Vision OCR ready."
