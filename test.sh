#!/usr/bin/sh

# inference example run
python src/test.py --seed 42 --config ./configs/MT/hyp.yaml --model timm-resnest50d --method dli-hh --data_perc 1.0 --poisson_prob 0.5 \
                   --load_weights ./output/perc_1.0/dli-hh/timm-resnest50d