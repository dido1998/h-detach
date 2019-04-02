# h-detach

This repository contains the code used to reproduce the paper [h-detach: Modifying the LSTM Gradient Towards Better Optimization](https://arxiv.org/abs/1810.03023). The final report can be found here- https://github.com/dido1998/iclr_2019/tree/master/papers/TeamYakataka-ryf7ioRqFX

## Software Requirements

Python 3, Pytorch 0.4, tensor board, tqdm

## Copying Task

First change the directory path for saving results on `line 27` of `copying.py`. 

To run copying task with time delay of `300` steps using h-detach with `0.5` probability of blocking gradinets through the h-state using seed value `3`, execute the following command:

`python copying.py --save-dir=enter_experiment_dir_name_here --p-detach=0.5 --T=300 --seed=3`

To run copying task without h-detach, execute the following command:

`python copying.py --save-dir=enter_experiment_dir_name_here --T=300 --seed=3`

To run transfer copying task using a previously trained model for T=200 (or any other value), run:

`python transfer_copy.py --T=200 --model-dir=enter_full_model_path`


## Sequential MNIST

First uncomment and change the directory path for saving results on `line 30` of `pixelmnist.py`.

To run pixel MNIST, execute,

`python pixelmnist.py --p-detach=0.25 --seed=250 --save-dir=enter_experiment_dir_name_here`

We experimented with multiple seeds (around 5), and picked the model with the best validation accuracy to report the final test performance. On our machine a seed of `400` gives the best validation model with test accuracy `98.62%`.

## Permuted MNIST

First uncomment and change the directory path for saving results on `line 30` of `pixelmnist.py`.

To run permuted pixel MNIST, execute,

`python pixelmnist.py --p-detach=0.25 --seed=150 --permute --save-dir=enter_experiment_dir_name_here`

We experimented with multiple seeds (around 5), and picked the model with the best validation accuracy to report the final test performance. On our machine a seed of `400` gives the best validation model with test accuracy `91.92%`.

## Image Captioning

The code for image captioning experiments can be found in the other branch called `captioning`.
