
## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following scripts in order to download the datasets and save them in our format:

```
python data/download_d4rl_datasets.py
```
to download the locomotion datasets, and
```
python data/download_antmaze.py
```
to download the antmaze datasets.

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py env=hopper dataset=medium model_type=rvs reweight_rtg=True conservative_percentile=0.95
```

The code currently supports `env={walker2d, hopper, halfcheetah}`, `dataset={medium-replay, medium, medium-expert}` for the locomotion tasks, and `env={umaze, medium, large}`, `dataset={play, diverse}` for the antmaze tasks.

The code implements three models `model_type={rvs, bc, dt}`. To turn off trajectory reweight or conservative objective, set `reweight_rtg=False` or `conservative_percentile=null`, respectively.

The experiments will be saved in exp