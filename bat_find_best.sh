#!/bin/bash
export EXPERIMENT_ID=386

全参数循环
for channel_index in 0 1 2 3 4 5
do
    for predict_timesteps in "120,168" "48,72" "3,6,12,24"
    do
        case $predict_timesteps in
        "120,168")
            input_predays_list=(7)
            base_timesteps_list=(24 48 120)
            ;;
        "48,72")
            input_predays_list=(3)
            base_timesteps_list=(24 48)
            ;;
        "3,6,12,24")
            input_predays_list=(1)
            base_timesteps_list=(24)
            ;;
        *)
            input_predays_list=(0 1)
            base_timesteps_list=(24 48)
            ;;
        esac
    
        for channel_roll in 0 1 2 3 4 5 6 7 8
        do
            for base_timesteps in "${base_timesteps_list[@]}"
            do
                for input_predays in "${input_predays_list[@]}"
                do
                    python /home/ming/_code/MingNet/AirNet6/train_model.py --epochs 100 --batch_size  24 --channel_index $channel_index --predict_timesteps $predict_timesteps --input_predays $input_predays --channel_roll $channel_roll --base_timesteps $base_timesteps  --model_layers 32,64,128 --data_dir /datapool/shared/AirNet --source_type guoKong --weather_source gfs_sx --model_dir sx_model_386 --train_date 2021-04-01 --test_date 2022-07-01 --end_date 2022-10-01 
                done
            done
        done
    done
done

