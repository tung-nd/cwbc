## Downloading datasets

Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

## Example usage

Scripts to reproduce our RvS results. Please replace `[DIRECTORY_NAME]` accordingly (e.g., `./dqn_replay`), and remove `--conservative_percentile` argument if training original RvS.

```
python run_dt_atari.py --game Seaquest --num_steps=500000 --epochs 5 --model_type=rvs --seed 123 --conservative_percentile 0.9 --data_dir_prefix [DIRECTORY_NAME]
```
