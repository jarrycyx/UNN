# Official Repository for "Ultra-efficient causal deep learning for Dynamic CSA-AKI Detection Using Minimal Variables"


[![medRxiv](https://img.shields.io/badge/Paper-medRxiv-blue)](https://www.medrxiv.org/content/10.1101/2023.12.04.23299332v1)
[![Colab](https://img.shields.io/badge/Code-Colab-orange)](https://colab.research.google.com/drive/1dPrgTSfqIWwUkGjaLjwNT1X925jz6n7P?usp=sharing)
[![Website](https://img.shields.io/badge/Website-Online-red)](http://www.causal-cardiac.com)

## ✍️ Paper summary

REACT (Real-time Evaluation and Anticipation with Causal disTillation): a causal deep learning approach that
combines the universal approximation abilities of neural networks with
causal discovery to develop REACT, a reliable and generalizable
model to predict a patient's risk of developing CSA-AKI within the next
48 hours.

## User-friendly website

Try dynamic early alerts of CSA-AKI at [web-based platform](http://www.causal-cardiac.com).

## Running the code on Google Colab

Run our example at [Google Colab](https://colab.research.google.com/drive/1dPrgTSfqIWwUkGjaLjwNT1X925jz6n7P?usp=sharing) to see how REACT works on simulated data.

## Running the code locally

### Clone the repository

```
git clone git@github.com:jarrycyx/UNN.git
cd UNN/REACT
```

### Setup the environment

```
conda create -n react_env python=3.8
conda activate react_env
pip install -r requirements.txt
```

### Run the notebook

You can run the notebook `run_example.ipynb` to see how REACT works on simulated data.


## 😘 Citation
If you use this code, please consider citing [our work](https://www.medrxiv.org/content/10.1101/2023.12.04.23299332v1).