#!/bin/bash
cd build/bin
./task2 -d ../../data/multiclass/train_labels.txt -m ../../model_multiclass.txt --train
cd ../..