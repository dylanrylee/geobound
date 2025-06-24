# GeoBound

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#)
[![DeepLab v3+](https://img.shields.io/badge/model-DeepLab%20v3%2B-green)](#)


## File Architecture
GeoBound/
├── data/               # data-loading & preprocessing scripts
├── notebooks/          # EDA, demos with Agriculture-Vision samples
├── src/
│   ├── dataset.py      # loader (SentinelHub + HF Datasets)
│   ├── model.py        # DeepLab v3+ definition & training loop
│   └── utils.py        # post-processing (morphology, contour extraction)
├── requirements.txt
├── README.md
