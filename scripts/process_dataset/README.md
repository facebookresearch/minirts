### Process Data

This folder contains code used to generate the dataset.

#### Generate Json

To generate a dataset for training from replays produced by game backend,
we first need to generate json files that contains all the information of
the game state at each time step:
```
python gen_state.py --replays-root PATH_TO_REPLAY_FOLDER \
       --player1 "dumper,fs=50" \
       --player2 "dummy,fs=50" \
       --human \
       --output-root OUTPUT_ROOT
```
Here, the `dumper,fs=50` for player1 means that we want to dump the state
for player1 every 50 frames while `dummpy,fs=50` means to ignore player2.
`--human` means that the replay is a result of a **human team play**, as described
in the paper regarding how we collect our data. Remove `--human` if the replay is
generated by two AIs playing against each other. For exmaple:
```
python gen_state.py --replays-root PATH_TO_REPLAY_FOLDER \
       --player1 "dumper,fs=50" \
       --player2 "dumper,fs=50" \
       --output-root OUTPUT_ROOT
```
This command will generate json for replays generated by two AI players, or 1 human vs 1 AI,
and it will dump state for both players.

#### Process Json

We need to process json file before individually before combining them into a dataset.
Simply run:
```
python process_json.py --src PATH_TO_JSON_FOLDER
```
This script will do some sanity checks, convert unit id to index, and handle unit action
targets properly.

#### Create Dataset

The last step is to create the dataset from processed json files.
```
python create_dataset.py \
       --raw_json_root PATH_TO_JSON_FOLDER \
       --processed_json_root PATH_TO_PROCESSED_JSON_FOLDER \
       --output PATH_TO_OUTPUT \
       --min_num_instruction 3 \
       --min_num_target 25
```
The `min_num_instruction` and `min_num_target` here use the value chosen in our paper to
remove some low quality games. They can be set to 0 if the replay is generated by AIs.