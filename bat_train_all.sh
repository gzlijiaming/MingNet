#!/bin/bash
export EXPERIMENT_ID=350
echo $EXPERIMENT_ID
# 最佳参数循环
# for channel_index in 0 1 2 3 4 5
for channel_index in 0 1 2 3 4 5
do
    python /home/ming/_code/MingNet/AirNet6/best_param_train.py --param_groud_id $EXPERIMENT_ID --epochs 100 --channel_index $channel_index --data_dir /datapool/shared/AirNet --source_type guoKong --weather_source gfs_sx --model_dir sx_model_350 --train_date 2021-04-01 --test_date 2022-07-01 --end_date 2022-10-01 
done


