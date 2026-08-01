#!/bin/sh
set -eu

CAMERA=/dev/diansai-camera
GRBL=/dev/diansai-grbl
SERVO=/dev/diansai-servo
ATTEMPTS=45

for attempt in $(seq 1 "$ATTEMPTS"); do
    if [ -e "$CAMERA" ] && [ -e "$GRBL" ] && [ -e "$SERVO" ]; then
        if /usr/bin/timeout 4 /usr/bin/v4l2-ctl --device="$CAMERA" \
            --stream-mmap=3 --stream-count=1 >/dev/null 2>&1; then
            exit 0
        fi
    fi
    /usr/bin/sleep 1
done

echo "Waiting for camera, GRBL, and servo USB devices timed out." >&2
exit 1
