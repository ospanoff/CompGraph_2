#!/bin/bash
cd build/bin
./task2 -d ../../data/multiclass/test_labels.txt -m ../../model_multiclass.txt -l predictions.txt --predict
cd ../..