#!/usr/bin/sh


# Table 1 - main - example run
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method baseline
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method focal
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method balanced
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method wce
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method dli-hh

# Fig 5 - weakly study - example run
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model mobileone_s1 --method dli-hh --poisson_prob 0.5 --data_perc 0.75
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model mobileone_s1 --method dli-hh --poisson_prob 0.5 --data_perc 0.50
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model mobileone_s1 --method dli-hh --poisson_prob 0.5 --data_perc 0.25
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model mobileone_s1 --method dli-hh --poisson_prob 0.5 --data_perc 0.10


# Table 2 - ablation study - example run
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method dli-cp --poisson_prob 0
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method dli-p --poisson_prob 1.0
python src/train.py --seed 42 --config ./configs/MT/hyp.yaml --model resnet18 --method dli-hh --poisson_prob 0.5

echo "Done"