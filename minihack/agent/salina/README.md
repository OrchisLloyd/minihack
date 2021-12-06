# salina_nethack

Install salina (i.e torch_agent)

```
> git clone https://github.com/facebookresearch/salina
> cd salina
> pip install -e .
```

## Runing a local exp

> OMP_NUM_THREADS=1 python salina_nethack/minihack_baselines/a2c/a2c_with_eval.py

It creates a "./outputs" directory with tensorboard outputs. Logs can also be read in python (see salina.logger)

## Running on SLURM

See the `.commands` files that give the command to laumch for grid search
