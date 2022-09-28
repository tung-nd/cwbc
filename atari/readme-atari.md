## Downloading and preprocessing datasets

Create a directory for the dataset and download the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

Run the following command to preprocess and save the data in our format:
```
python create_dataset.py --game [GAME_NAME] --data_dir_prefix [DQN_REPLAY_DIRECTORY] --save_dir [SAVE_DIRECTORY]
```
in which `[DQN_REPLAY_DIRECTORY]` is the path to the original data directory, e.g., `./dqn_replay`, and `[SAVE_DIRECTORY]` is the path to the data directory you want to save to, e.g., `./data`.

## Example usage

Scripts to reproduce our RvS results. Please replace `[ROOT_DIR]` and `[DATA_DIR]` accordingly (e.g., `./exp/cwbc` and `./data`).

```
python run_atari.py --root [ROOT_DIR] --data_dir [DATA_DIR] --game Qbert --model_type=rvs --avg_reward --reweight_rtg --bins 20 --percentile 0.5 --lamb 0.1 --conservative_percentile 0.95 --conservative_std 500.0 --conservative_w 0.1
```
