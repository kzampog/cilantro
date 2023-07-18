#!/usr/bin/env bash
# This is meant to be run from the repo root.
find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -not -path "./include/cilantro/3rd_party/*" -not -path "./src/3rd_party/*" -not -path "./build/*" -exec clang-format -i {} \;
