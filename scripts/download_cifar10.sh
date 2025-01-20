#!/bin/bash

wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P data
tar -xzvf data/cifar-10-binary.tar.gz
rm data/cifar-10-binary.tar.gz
mv cifar-10-batches-bin data/cifar-10
