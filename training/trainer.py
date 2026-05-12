

import torch
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device, max_grad_norm=float('inf'), checkpoint_interval=2500, checkpoint_dir = None, scheduler=None, epochs=1, max_patience=5, diag_batch=None, diag_interval=200, compute_update_norm=False):
   
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.max_patience = max_patience
        self.scheduler = scheduler
        self.diag_batch = diag_batch
        self.diag_interval = diag_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.max_grad_norm = max_grad_norm
        self.compute_update_norm = compute_update_norm
        self.pred_snapshots = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        if self.compute_update_norm:
            with torch.no_grad():
                params_before = [p.clone() for p in self.model.parameters()]

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            total_grad_norm += grad_norm.item()
            self.optimizer.step()
            total_loss += loss.item()

        update_norm = 0.0
        if self.compute_update_norm:
            with torch.no_grad():
                update_norm = torch.sqrt(sum((p - pb).norm()**2 for p, pb in zip(self.model.parameters(), params_before))).item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, total_grad_norm, update_norm
    
    def evaluate(self):
        self.model.eval()
        loss_total = 0.0
        avg_loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device) 
                output = self.model(x)
                loss = self.loss_fn(output,y)
                loss_total += loss.item()

            loss_avg = loss_total/len(self.val_loader)

        return loss_avg

    def snapshot(self, epoch):
        self.model.eval()
        with torch.no_grad():
            x_d, y_d = self.diag_batch
            preds = self.model(x_d).cpu().numpy()
            y_d_np = y_d.cpu().numpy()
        residuals = preds - y_d_np
        r2 = float(1 - np.var(residuals) / np.var(y_d_np))
        self.pred_snapshots.append({
            "epoch": epoch,
            "preds": preds,
            "y_true": y_d_np,
            "mean_residual": float(residuals.mean()),
            "std_residual": float(residuals.std()),
            "r2": r2,
        })
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"r2_snapshot": r2}, step=epoch)

    def train(self):
        train_losses, val_losses, total_grad_norms, total_update_norms = [], [], [], []    
        patience_counter = 0
        best_val_loss = float('inf')
        best_state = None
        logged_beat_baseline = False
        for epoch in range(self.epochs):

            avg_train_loss, total_grad_norm, total_update_norm = self.train_one_epoch()
            avg_val_loss = self.evaluate()

            if self.diag_batch is not None and self.diag_interval and epoch % self.diag_interval == 0:
                self.snapshot(epoch)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            total_grad_norms.append(total_grad_norm)
            total_update_norms.append(total_update_norm)

            if _WANDB_AVAILABLE and wandb.run is not None:
                log_dict = {
                    "losses/train": avg_train_loss,
                    "losses/val": avg_val_loss,
                    "grad_norm": total_grad_norm,
                    "update_norm": total_update_norm,
                }
                mse_1nn = wandb.config.get("mse_1nn", None)
                if mse_1nn is not None:
                    log_dict["losses/mse_1nn"] = mse_1nn
                    if avg_val_loss < mse_1nn and not logged_beat_baseline:
                        log_dict["beat_baseline_epoch"] = epoch
                        logged_beat_baseline = True
                wandb.log(log_dict, step=epoch)

            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}"
            )

            if avg_val_loss < best_val_loss:    
                best_val_loss = avg_val_loss  
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()    


            if self.checkpoint_dir is not None and self.checkpoint_interval and epoch % self.checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'total_grad_norms': total_grad_norms,
                    'total_update_norms': total_update_norms
                }, self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt")
       
        if best_state is not None:
            self.model.load_state_dict(best_state)

       

        return train_losses, val_losses, total_grad_norms, total_update_norms