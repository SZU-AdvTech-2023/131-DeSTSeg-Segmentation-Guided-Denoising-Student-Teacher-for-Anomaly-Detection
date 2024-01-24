#!/usr/bin/env bash

mkdir datasets
cd datasets
# Download describable textures dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

