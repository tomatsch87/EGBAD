import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import wandb
import torchvision
import pathlib
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt

from x_ray_dataloader import XRayDataset
from egbad import EGBAD

def log_generations(model, data, epoch):
    model.G.eval()
    model.E.eval()
    model.D.eval()
    
    with torch.no_grad():
        generated_images = model.generate_samples(8)
        generated = generated_images.cpu().numpy()
        real_batch = next(iter(data))
        real_batch = real_batch.to(model.device)
        real_subset = real_batch[:8]

        encoded_latents = model.E(real_subset)
        reconstructed_images = model.G(encoded_latents)
        real_imgs = real_subset.cpu().numpy()
        recon_imgs = reconstructed_images.cpu().numpy()
        
        generated_image_list = []
        for i in range(8):
            gen_img = (generated[i, 0] + 1.0) / 2.0
            _, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(gen_img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Generated Sample {i+1}')
            ax.axis('off')
            plt.tight_layout()
            generated_image_list.append(wandb.Image(plt, caption=f"generated_sample_{i}"))
            plt.close()
        
        reconstruction_image_list = []
        for i in range(8):
            orig_img = (real_imgs[i, 0] + 1.0) / 2.0
            recon_img = (recon_imgs[i, 0] + 1.0) / 2.0
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(orig_img, cmap='gray', vmin=0, vmax=1)
            ax1.set_title('Original x')
            ax1.axis('off')
            ax2.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('Reconstruction G(E(x))')
            ax2.axis('off')
            plt.tight_layout()
            reconstruction_image_list.append(wandb.Image(plt, caption=f"reconstruction_{i}"))
            plt.close()
    
    wandb.log({
        "generated_samples": generated_image_list,
        "reconstructions": reconstruction_image_list
    }, step=epoch)


def train_step(
    model, healthy_data, epoch, device, pneumonia_val, **kwargs
):
    model.G.train()
    model.E.train()
    model.D.train()
    ge_loss_mean = 0
    d_loss_mean = 0
    recon_loss_mean = 0

    for healthy in tqdm(healthy_data, desc=f"Training Epoch {epoch}"):
        healthy = healthy.to(device)

        d_loss, ge_loss, recon_loss = model.train_step(healthy)

        ge_loss_mean += ge_loss
        d_loss_mean += d_loss
        recon_loss_mean += recon_loss

    train_ge_loss = ge_loss_mean / len(healthy_data)
    train_d_loss = d_loss_mean / len(healthy_data)
    recon_loss_mean = recon_loss_mean / len(healthy_data)
    print(f"Epoch {epoch}: Train GE Loss: {train_ge_loss}, Train D Loss: {train_d_loss}, Train Recon Loss: {recon_loss_mean}")

    # Validate
    model.G.eval()
    model.E.eval()
    model.D.eval()
    ge_val_loss_mean = 0
    d_val_loss_mean = 0
    recon_val_loss_mean = 0
    with torch.no_grad():
        for pneumonia in tqdm(pneumonia_val, desc=f"Validation Epoch {epoch}"):
            pneumonia = pneumonia.to(device)

            d_loss, ge_loss, recon_loss = model.validate_step(pneumonia)
            ge_val_loss_mean += ge_loss
            d_val_loss_mean += d_loss
            recon_val_loss_mean += recon_loss

    val_ge_loss = ge_val_loss_mean / len(pneumonia_val)
    val_d_loss = d_val_loss_mean / len(pneumonia_val)
    val_recon_loss = recon_val_loss_mean / len(pneumonia_val)
    print(f"Epoch {epoch}: Val GE Loss: {val_ge_loss}, Val D Loss: {val_d_loss}, Val Recon Loss: {val_recon_loss}")

    wandb.log({
        "train_ge_loss": train_ge_loss,
        "train_d_loss": train_d_loss,
        "train_recon_loss": recon_loss_mean,
        "val_ge_loss": val_ge_loss,
        "val_d_loss": val_d_loss,
        "val_recon_loss": val_recon_loss
    }, step=epoch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def anomaly_detection(
    model, pneumonia_data, normal_data, alpha, epoch, device, **kwargs
):
    model.D.eval()
    model.G.eval()
    model.E.eval()
    anomaly_label = 1
    normal_label = 0

    pneumonia_scores = []
    normal_scores = []
    with torch.no_grad():
        for data in tqdm(pneumonia_data, desc="Processing pneumonia samples"):
            data = data.to(device)

            # Anomaly score based on the EGBAD paper (see http://arxiv.org/abs/1802.06222)

            # Compute Anomaly Score: A(x) = aLG(x) + (1 − a)LD(x)
            # LG(x) = ||x − G(E(x))||1

            # Using the second variant (feature-matching loss) described in the EGBAD paper for LD(x)
            # LD(x) = ||fD(x, E(x)) − fD(G(E(x)), E(x))||1

            DE = model.E(data)
            DG = model.G(DE)
            _, real_inter  = model.D(data, DE)
            _, gen_inter = model.D(DG, DE)

            for i in range(data.size(0)):
                l_g = nn.L1Loss(reduction="sum")(data[i], DG[i])
                l_d = nn.L1Loss(reduction="sum")(real_inter[i], gen_inter[i])
                anomaly_score = alpha * l_g + (1 - alpha) * l_d
                pneumonia_scores.append(anomaly_score.item())

        for data in tqdm(normal_data, desc="Processing normal samples"):
            data = data.to(device)

            # Compute Anomaly Score: A(x) = aLG(x) + (1 − a)LD(x)
            # LG(x) = ||x − G(E(x))||1
            # LD(x) = ||fD(x, E(x)) − fD(G(E(x)), E(x))||1

            DE = model.E(data)
            DG = model.G(DE)
            _, real_inter  = model.D(data, DE)
            _, gen_inter = model.D(DG, DE)

            for i in range(data.size(0)):
                l_g = nn.L1Loss(reduction="sum")(data[i], DG[i])
                l_d = nn.L1Loss(reduction="sum")(real_inter[i], gen_inter[i])
                anomaly_score = alpha * l_g + (1 - alpha) * l_d
                normal_scores.append(anomaly_score.item())

        all_scores = pneumonia_scores + normal_scores
        all_scores = np.array(all_scores)
        all_labels = [normal_label] * len(pneumonia_scores) + [anomaly_label] * len(normal_scores)
        all_labels = np.array(all_labels)

        # Calculate anomaly detection threshold
        threshold = (np.mean(normal_scores) + np.mean(pneumonia_scores)) / 2
        model.anomaly_threshold = threshold

        roc_auc_score_value = roc_auc_score(
            y_true=all_labels,
            y_score=all_scores,
        )

        # Predict anomalies
        y_pred = (all_scores > model.anomaly_threshold).astype(int)

        best_f1 = f1_score(
            y_true=all_labels,
            y_pred=y_pred
        )
        phi_score = matthews_corrcoef(
            y_true=all_labels,
            y_pred=y_pred,
        )

        print(f"Epoch {epoch}: Best F1 Score: {best_f1}, ROC AUC Score: {roc_auc_score_value}, Phi Score: {phi_score}")
        
        wandb.log({
            "f1_score": best_f1,
            "roc_auc_score": roc_auc_score_value,
            "phi_score": phi_score,
            "anomaly_score_mean": np.mean(pneumonia_scores),
            "normal_score_mean": np.mean(normal_scores),
            "threshold": model.anomaly_threshold,
        }, step=epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_f1


def train_with_config():
    model = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with wandb.init():
            config = wandb.config
            device = "cuda" if not config.cpu else "cpu"
            
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
                ]
            )
            
            pneumonia_data = XRayDataset(
                path=pathlib.Path("/data/chest_x_ray/train/pneumonia"),
                transform=transforms,
            )
            normal_data = XRayDataset(
                path=pathlib.Path("/data/chest_x_ray/train/normal"),
                transform=transforms,
            )
            pneumonia_val = XRayDataset(
                path=pathlib.Path("/data/chest_x_ray/test/pneumonia"),
                transform=transforms,
            )
            pneumonia_data = torch.utils.data.DataLoader(
                pneumonia_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            normal_data = torch.utils.data.DataLoader(
                normal_data,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            pneumonia_val = torch.utils.data.DataLoader(
                pneumonia_val,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            model = EGBAD(filter_size=config.filter_size, g_lr=config.g_learning_rate, e_lr=config.e_learning_rate, d_lr=config.d_learning_rate, betas=config.betas, device=device, latent_dim=config.latent_dim, anomaly_threshold=0.5)

            # Early stopping variables
            patience = 30
            best_f1 = 0.0
            f1_score = 0.0
            f1_patience = 0

            for e in range(config.num_epochs):
                train_step(
                    model=model,
                    healthy_data=pneumonia_data,
                    epoch=e,
                    device=device,
                    pneumonia_val=pneumonia_val,
                )

                f1_score = anomaly_detection(
                    model=model,
                    pneumonia_data=pneumonia_val,
                    normal_data=normal_data,
                    alpha=config.alpha,
                    epoch=e,
                    device=device,
                )

                # Early stopping based on F1 score
                if f1_score is not None and (best_f1 is None or f1_score > best_f1):
                    best_f1 = f1_score
                    f1_patience = 0

                    model_name = f"model_e{e}_f1{best_f1:.5f}_fs{config.filter_size}_ld{config.latent_dim}"
                    model_checkpoint = f"x_ray_data/output/{model_name}.pth.tar"
                    model.save_models(checkpoint_path=model_checkpoint)
                    print(f"Best model saved at epoch {e} with F1 score: {best_f1:.5f}")
                else:
                    f1_patience += 1

                # Early stopping if no improvement for patience epochs
                if f1_patience >= patience:
                    print(f"Early stopping at epoch {e}. F1 score hasn't improved for {f1_patience} epochs.")
                    wandb.log({"early_stopped_epoch": e}, step=e)
                    break

                # Update learning rate schedulers after 200 epochs
                if e > 200:
                    model.update_schedulers()

                log_generations(model, pneumonia_data, e)

    except Exception as e:
        print(f"Error in training: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model is not None:
            del model


def setup_sweep():
    sweep_config = {
        'name': 'sweep_1',
        'method': 'random',
        'metric': {'name': 'f1_score', 'goal': 'maximize'},
        'parameters': {
            'd_learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 7e-4
            },
            'g_learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-4,
                'max': 1e-3
            },
            'e_learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-4,
                'max': 1e-3
            },
            'batch_size': {
                'value': 100
            },
            'filter_size': {
                'value': 64
            },
            'latent_dim': {
                'values': [200, 256, 512]
            },
            'num_epochs': {
                'value': 400
            },
            'alpha': {
                'values': [0.1, 0.25, 0.5, 0.75, 0.9]
            },
            'betas': {
                'value': (0.5, 0.999)
            },
            'lambda_gp': {
                'value': 10
            },
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="xray_egbad_sweep")
    wandb.agent(sweep_id, train_with_config, count=10)

if __name__ == "__main__":
    wandb.login()
    setup_sweep()
    print("Hyperparameter sweep completed.")
