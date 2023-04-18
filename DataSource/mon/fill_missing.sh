#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/../../

~/anaconda3/bin/conda run -n air python ${SCRIPT_DIR}/production.py --data_dir /datapool/shared/production --source_type guoKong --start_date 2022-09-01 --db_config ~/.config/mon-db.json --db mondb --lookback_days 16 