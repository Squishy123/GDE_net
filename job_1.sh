#!/bin/bash
python src/moving_mnist/generate_encoded_cache.py -f 1 -o "mnist_preloaded_encoded_1.npz" ; 
python src/moving_mnist/sae_training.py ;
python src/moving_mnist/generate_encoded_cache.py -f 10 -o "mnist_preloaded_encoded_10.npz" ;
python src/moving_mnist/generate_encoded_cache.py -f 20 -o "mnist_preloaded_encoded_20.npz" ; 


