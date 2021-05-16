#!/bin/bash
echo "Downloading OPUS... "
opus_get -s ro -p raw -q -l | grep ro.zip > opus_list.txt
python opus_ro_get.py -f opus_list.txt -o ./raw/opus
rm opus_list.txt