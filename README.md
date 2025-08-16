# EGBAD - Efficient GAN-Based Anomaly Detection

## Overview

For the task of anomaly detection on a dataset of chest X-ray images, this repository features an implementation and evaluation of the `EGBAD` [Efficient GAN-Based Anomaly Detection](http://arxiv.org/abs/1802.06222) model. Additionally, various enhancements to improve its performance have been implemented, including a `WGAN` (Wasserstein GAN) version of EGBAD with gradient penalty.

## EGBAD Adaptations

`EGBAD` utilizes a `BiGAN` [Bidirectional Generative Adversarial Network](http://arxiv.org/abs/1605.09782) with an additional anomaly score based on feature and pixel-valued reconstruction to tackle the task of anomaly detection.

Additionally, some ideas from "An improved BiGAN based approach for anomaly detection" by [Kaplan and Alptekin](https://www.sciencedirect.com/science/article/pii/S1877050920318445) were implemented with an additional reconstruction loss term to the BiGAN to stabilize and enhance the training process. The best variant turned out to be a feature-based reconstruction loss that jointly trains Encoder and Generator, which significantly improved the model's performance over the original EGBAD implementation.

Lastly, a `WGAN` (Wasserstein GAN) version of EGBAD including a gradient penalty term from "Improved Training of Wasserstein GANs" by [Gulrajani et al.](http://arxiv.org/abs/1704.00028) was implemented, which stabilized the training process significantly, but was not able to further improve the performance of the model on the X-ray dataset.

## Dataset

The models in this repository are trained and evaluated on the [chest X-ray image](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, focusing on distinguishing between normal (unhealthy) and anomalous (healthy) chest X-ray images.

## Repository Structure

```
EGBAD/
├── egbad.py - The adapted EGBAD model implementation
├── eval.py - Evaluation script
├── main.py - Main training script including hyperparameter optimization and logging to wandb
├── test_evaluation_results.json - Evaluation results on the test dataset
├── wgan_egbad.py - WGAN implementation of EGBAD
├── x_ray_dataloader.py - Data loader for the X-ray dataset
├── .gitignore
└── README.md
```

## Usage

```bash
git clone https://github.com/tomatsch87/EGBAD.git
cd EGBAD
```

### Training

There is no `requirements.txt` file provided, please ensure you have the necessary libraries installed. Then proceed to train the model with:

```bash
python main.py
```

### Evaluation

You need to have a trained model to evaluate. Copy the path to your saved trained model and adjust the `eval.py` script accordingly. The evaluation script will compute the best f1_score, roc_auc and mcc score for this model on the test dataset:

```bash
python eval.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
