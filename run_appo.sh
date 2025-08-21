python -m scripts.main --algo APPO --num_workers 24 --num_envs_per_worker 20 \
                       --batch_size 4096 --reward_scale 0.1 --obs_scale 255.0 \
                       --train_for_env_steps 5_000_000_000 --save_every_steps 6_000_000_000 \
                       --code_seed 42 --seed 2  --stats_avg 1000 \
                       --train_dir train_dir/skill_policy/ --experiment default \
                       --evaluation True --eval_target goldenexit 
