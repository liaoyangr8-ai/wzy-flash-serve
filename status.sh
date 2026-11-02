#!/usr/bin/env bash
# status.sh -- Inspect wzy-flash-serve health, GPUs, and recent requests
curl -s http://127.0.0.1:11438/metrics | python3 -m json.tool | head -n 60
