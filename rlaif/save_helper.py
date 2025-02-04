import csv
import os

import torch

from sample_factory.algorithms.appo.learner import LearnerWorker


def save(
    cfg, 
    tot_num_iter,
    reward_model,
    optimizer,
    train_met,
    val_met,
    full_reward_rms,
    all_msgs_rewards,
    all_msgs_input
):
    print(f'Saving at iter {tot_num_iter}...')
    exp_path = os.path.join(cfg.train_dir, cfg.experiment)

    # Saving train satistics
    with open(f'{exp_path}/train_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = list(train_met.keys())
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        num_rows = len(train_met[fieldnames[0]])
        for i in range(num_rows):
            row_data = [train_met[key][i] for key in fieldnames]
            writer.writerow(row_data)

    with open(f'{exp_path}/val_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = list(val_met.keys())
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        num_rows = len(val_met[fieldnames[0]])
        for i in range(num_rows):
            row_data = [val_met[key][i] for key in fieldnames]
            writer.writerow(row_data)

    # Saving checkpoint
    checkpoint = {
        'tot_num_iter': tot_num_iter,
        'model': reward_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'reward_mean': full_reward_rms.mean[0],
        'reward_var': full_reward_rms.var[0],
    }
    checkpoint_dir = LearnerWorker.checkpoint_dir(cfg, policy_id=0)
    tmp_filepath = os.path.join(checkpoint_dir, 'temp_checkpoint.pth')
    checkpoint_name = f'checkpoint_{tot_num_iter}.pth'
    filepath = os.path.join(checkpoint_dir, checkpoint_name)

    print(f'Saving to {filepath}...')
    torch.save(checkpoint, tmp_filepath)
    os.rename(tmp_filepath, filepath)

    metrics_folder = f'{exp_path}/reward_metrics'
    os.makedirs(metrics_folder, exist_ok=True)

    # Save quantiles of the reward function .
    # This will be used in the RL training loop.
    all_msgs_rewards_norm = (all_msgs_rewards - full_reward_rms.mean
                             ) / torch.sqrt(full_reward_rms.var)
    quantiles = [i / 100 for i in range(5, 96, 5)]

    rew_norm_quantiles = []
    for quantile in quantiles: 
        rew_norm_quantiles.append(
            torch.quantile(all_msgs_rewards_norm, quantile).item()
        )
    rew_norm_quantiles = [f'{q:.2f}' for q in rew_norm_quantiles]
    csv_file = f'{metrics_folder}/train_norm_quantiles.csv'
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if tot_num_iter == 0:
            writer.writerow(quantiles)
        writer.writerow(rew_norm_quantiles)
