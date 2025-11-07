import numpy as np
import torch # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore


def evaluate_loader(model, loader, device, criterion):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss, total_mae, n_batches = 0.0, 0.0, 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item()
            total_mae += (pred - yb).abs().mean().item()
            n_batches += 1
    
    return total_loss / n_batches, total_mae / n_batches


def check_early_stopping(epoch, val_mae, best_val, wait, patience, model, best_state, best_epoch, start_after=20, min_delta=1e-6):
    """Early stopping with warmup. Tracks best from the beginning, enforces patience after `start_after`."""
    improved = val_mae < best_val - min_delta
    if improved:
        best_val = val_mae
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        wait = 0

    # During warmup epochs, never stop
    if epoch <= start_after:
        return False, best_val, best_state, best_epoch, wait

    # After warmup, count patience only when not improving
    if not improved:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            return True, best_val, best_state, best_epoch, wait

    return False, best_val, best_state, best_epoch, wait


def subject_mae(model, train_seg_subjects, X_train, y_train, device):
    print("\nPer-subject Training MAE:")
    model.eval()
    for s in np.unique(train_seg_subjects):
        idx = np.where(train_seg_subjects == s)[0]
        if len(idx) == 0:
            continue
        
        X_s = torch.from_numpy(X_train[idx].astype(np.float32))
        y_s = torch.from_numpy(y_train[idx].astype(np.float32))
        
        ds = TensorDataset(X_s, y_s)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            preds, gts = [], []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu())
                gts.append(yb.cpu())
            
            pred_all = torch.cat(preds, dim=0)
            gt_all = torch.cat(gts, dim=0)
            mae = torch.mean(torch.abs(pred_all - gt_all)).item()
            
            print(f"  Subject {int(s)}: MAE = {mae:.5f} ({len(idx)} segments)")