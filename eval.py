import json
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import pathlib
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import numpy as np
from x_ray_dataloader import XRayDataset
from egbad import EGBAD


def load_model(model_path, device, filter_size=64, latent_dim=200):
    model = EGBAD(filter_size=filter_size, latent_dim=latent_dim, device=device)
    model.load_models(model_path)
    model.G.eval()
    model.E.eval()
    model.D.eval()
    print(f"EGBAD model loaded from {model_path}")
    return model


def evaluate_model(model, alpha, pneumonia_test, normal_test, device):
    model.G.eval()
    model.E.eval()
    model.D.eval()
    
    normal_label = 0  # pneumonia images
    anomaly_label = 1  # normal images
    
    pneumonia_scores = []
    normal_scores = []

    # Process test data
    with torch.no_grad():
        for data in tqdm(pneumonia_test, desc="Processing pneumonia samples"):
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
                pneumonia_scores.append(anomaly_score.item())

        for data in tqdm(normal_test, desc="Processing normal samples"):
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
    
    # Combine all scores and labels
    all_scores = pneumonia_scores + normal_scores
    all_scores = np.array(all_scores)
    all_labels = [normal_label] * len(pneumonia_scores) + [anomaly_label] * len(normal_scores)
    all_labels = np.array(all_labels)
    
    print(f"\nDiscriminator Score Statistics:")
    print(f"Pneumonia (normal) - Mean: {np.mean(pneumonia_scores):.4f}, Std: {np.std(pneumonia_scores):.4f}")
    print(f"Normal (anomaly) - Mean: {np.mean(normal_scores):.4f}, Std: {np.std(normal_scores):.4f}")
    print(f"Overall - Mean: {np.mean(all_scores):.4f}, Std: {np.std(all_scores):.4f}, Max: {np.max(all_scores):.4f}")
    
    # Exhaustive search for optimal threshold
    min_loss = np.min(all_scores)
    max_loss = np.max(all_scores)
    threshold_candidates = np.linspace(min_loss, max_loss, 1000)
    
    best_f1 = 0
    best_threshold_exhaustive = None
    
    print("Performing exhaustive threshold search...")
    for threshold in tqdm(threshold_candidates, desc="Searching optimal threshold"):
        y_pred = (all_scores > threshold).astype(int)
        try:
            f1 = f1_score(y_true=all_labels, y_pred=y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_exhaustive = threshold
        except (ValueError, ZeroDivisionError):
            continue
    
    print(f"Best threshold found: {best_threshold_exhaustive:.4f} with F1 score: {best_f1:.4f}")
    
    # Calculate optimal threshold using multiple methods
    thresholds = {
        'mean': (np.mean(pneumonia_scores) + np.mean(normal_scores)) / 2,
        'median': (np.median(pneumonia_scores) + np.median(normal_scores)) / 2,
        'percentile_95': np.percentile(pneumonia_scores, 95),
        'exhaustive_search': best_threshold_exhaustive
    }
    
    results = {}
    
    for threshold_name, threshold in thresholds.items():
        print(f"\n=== Results with {threshold_name} threshold ({threshold:.4f}) ===")

        # Predict labels based on threshold
        y_pred = (all_scores > threshold).astype(int)
        
        try:
            roc_auc = roc_auc_score(y_true=all_labels, y_score=all_scores)
            f1 = f1_score(y_true=all_labels, y_pred=y_pred)
            mcc = matthews_corrcoef(y_true=all_labels, y_pred=y_pred)
        
            
            results[threshold_name] = {
                'threshold': threshold,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'mcc': mcc,
            }
            
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"MCC (Phi): {mcc:.4f}")

            cm = confusion_matrix(y_true=all_labels, y_pred=y_pred)
            print(f"\n=== Confusion Matrix ===")
            print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not calculate metrics for {threshold_name}: {e}")
            results[threshold_name] = None
    
    # Find best threshold based on F1 score
    best_threshold_name = max([k for k, v in results.items() if v is not None], 
                             key=lambda k: results[k]['f1_score'])
    best_results = results[best_threshold_name]
    
    print(f"\n=== BEST RESULTS ({best_threshold_name} threshold) ===")
    for metric, value in best_results.items():
        print(f"{metric}: {value:.4f}")
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    alpha = 0.75
    filter_size = 64
    latent_dim = 200
    
    # Model path
    model_path = "x_ray_data/output/model_e263_f10.96403_fs64_ld200.pth.tar"
    
    model = load_model(model_path, device, filter_size, latent_dim)
    transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,))
                ]
            )

    print("Loading test datasets...")
    pneumonia_test_dataset = XRayDataset(
        path=pathlib.Path("/data/chest_x_ray/test/pneumonia"),
        transform=transforms,
    )
    
    normal_test_dataset = XRayDataset(
        path=pathlib.Path("/data/chest_x_ray/test/normal"),
        transform=transforms,
    )
    
    batch_size = 100
    pneumonia_test_loader = torch.utils.data.DataLoader(
        pneumonia_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    normal_test_loader = torch.utils.data.DataLoader(
        normal_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Test data loaded:")
    print(f"Pneumonia test samples: {len(pneumonia_test_dataset)}")
    print(f"Normal test samples: {len(normal_test_dataset)}")
    
    print("\nStarting evaluation...")
    results = evaluate_model(
        model, alpha, pneumonia_test_loader, normal_test_loader, device
    )
    
    results_file = "test_evaluation_results.json"
    with open(results_file, 'w') as f:
        json_results = {}
        for k, v in results.items():
            if v is not None:
                json_results[k] = {metric: float(value) for metric, value in v.items()}
            else:
                json_results[k] = None
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
