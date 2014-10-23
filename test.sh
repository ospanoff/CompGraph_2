#!/bin/bash
cd build/bin
./task2 -d ../../data/binary/test_labels.txt -m ../../model_binary.txt -l predictions.txt --predict
cd ../..