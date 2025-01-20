#!/bin/bash

xmake clean && xmake -y
xmake run main --config=../../../../examples/mnist.json --device=gpu --phase=train
