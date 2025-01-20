#########################################################################
# File Name:    scripts/download_cifar10.sh
# Author:       林毅波
# mail:         linyibo_2024@163.com
# Created Time: Mon 20 Jan 2025 01:04:41 PM CST
#########################################################################
#!/bin/bash

wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P data
tar -xzvf data/cifar-10-binary.tar.gz
rm data/cifar-10-binary.tar.gz
mv cifar-10-batches-bin data/cifar-10
