#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/../../

~/anaconda3/bin/conda run -n air python ${SCRIPT_DIR}/production.py --data_dir /datapool/shared/production --source_type guoKong --weather_source gfs_sx --start_date 2022-10-01 --db_config ~/.config/mon-db.json --db mondb --model_dir model350 --param_groud_id 350 --lookback_days 10 
