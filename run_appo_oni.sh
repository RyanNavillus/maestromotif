python -m scripts.main --train_dir ./runs \
    --experiment default \
    --root_env NetHackScoreExtendedActions-v1 \
    --llm_reward_coeff 0.0 \
    --extrinsic_reward_coeff 0.1 \
    --wandb True \
    --wandb_entity ryansullivan \
    --wandb_proj syllabus-testing
