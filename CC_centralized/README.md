## Flower Task Repository
This repository contains code for Centralized and federated learning experiments using PyTorch and FLWR, focusing on centralized and federated training workflows.

## Table of Contents
    Project Overview

    Files Description

    Getting Started

    Running the Code

    Contributing

    License

    Contact

## Project Overview
This project implements both centralized and federated learning workflows, including centralized training and federated training setups, leveraging PyTorch and FLWR frameworks. It is designed to explore GANs and client implementations in a federated setting.

## Files Description
`* Open the terminal`
`* centralized_fashion.py`
Script to run centralized training experiments on the Fashion dataset.

`* Open another terminal`
`* run_training.py`
Main entry point for federated training, orchestrating the overall training process.

config_centralized.py
Configuration file containing parameters and settings for centralized training.

train_centralized.py
Contains the training logic and functions used during centralized training.

Running the Code
Centralized Training
To run centralized training on the Fashion dataset, execute:

bash
python centralized_fashion.py
This script uses config_centralized.py for configuration and train_centralized.py for training logic.

Federated Training
To run federated training, execute:

bash
python run_training.py
This script coordinates the federated learning workflow using FLWR.

Contributing
Contributions are welcome! Please fork the repository and submit pull requests with clear descriptions of your changes.