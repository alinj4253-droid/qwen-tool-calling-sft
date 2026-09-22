#!/bin/bash
set +e
for p in $(pgrep -f "fetch_datasets.py"); do kill "$p" 2>/dev/null && echo "killed fetch $p"; done
for p in $(pgrep -f "aria2c.*datasets_raw"); do kill "$p" 2>/dev/null && echo "killed aria $p"; done
sleep 2
for p in $(pgrep -f "aria2c.*datasets_raw"); do kill -9 "$p" 2>/dev/null; done
echo done
