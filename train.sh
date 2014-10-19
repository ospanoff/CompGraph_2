#!/bin/bash
cd build/bin
./task2 -d ../../data/binary/train_labels.txt -m model.txt --train
cd ../..