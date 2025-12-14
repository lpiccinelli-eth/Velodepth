# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Download the processed datasets from [Huggingface Datasets](https://huggingface.co/datasets/lpiccinelli/velodepth-evaluation) and put them in your `$DATAROOT` directory, using `huggingface-cli`:

```bash
export DATAROOT=$HOME/data/eval
huggingface-cli download lpiccinelli/velodepth --repo-type dataset --local-dir $DATAROOT --local-dir-use-symlinks False
```

## Configuration

See [`configs/velodepth-eval.json`](../configs/velodepth-eval.json) for an example of evaluation configurations on all benchmarks. You can modify "data/val_datasets" to modify the testing dataset list.


## Run Evaluation

Run the script [`scripts/eval.py`](../scripts/eval.py):

```bash
# Evaluate VeloDepth on the benchmarks
python scripts/eval.py --dataroot $DATAROOT --config-file configs/velodepth-eval.json --save-path ./velodepth-results.json --camera-gt
```


With arguments:

```bash
Usage: eval.py [OPTIONS]

  Evaluation script.

Options:
  --config-file PATH    Path to the evaluation configurations.
  --dataroot PATH  Path to the where the hdf5 datasets are stored
  --save-path PATH Path to the output json file.
  --camera-gt      Use camera-gt during evaluation.
```
