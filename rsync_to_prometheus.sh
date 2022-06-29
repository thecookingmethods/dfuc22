#!/bin/bash
echo "uploading to $1"
rsync \
-a \
--exclude='.*' --exclude='*.h5' --exclude='venv' --exclude='rsync_to_prometheus.sh' \
./ plgdarekk@prometheus.cyfronet.pl:/net/people/plgdarekk/dfuc/$1 \
--verbose