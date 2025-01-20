#!/bin/bash

xmake clean && xmake -y
xmake run main --config=../../../../examples/cifar-10.json --device=gpu --phase=train
