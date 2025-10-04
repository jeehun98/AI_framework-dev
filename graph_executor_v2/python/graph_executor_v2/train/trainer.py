# graph_executor_v2/train/trainer.py
from typing import Optional, Dict, Any, Iterable, Callable

class Callback:
    def on_epoch_start(self, **kw): pass
    def on_epoch_end(self, **kw): pass
    def on_batch_end(self, **kw): pass

class CSVLogger(Callback):
    def __init__(self, path="train_log.csv"): self.path = path; open(self.path,"w").write("epoch,step,loss\n")
    def on_batch_end(self, epoch, step, loss, **kw):
        with open(self.path, "a") as f: f.write(f"{epoch},{step},{loss:.6f}\n")

class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler=None, callbacks: Optional[list[Callback]]=None):
        self.model = model; self.optimizer = optimizer
        self.loss_fn = loss_fn; self.scheduler = scheduler
        self.cbs = callbacks or []

    def train_epoch(self, dataloader, epoch: int, grad_clip: Optional[float]=None):
        total = 0.0; steps = 0
        for step, (x, y) in enumerate(dataloader):
            y_pred = self.model(x)
            loss, dyp = self.loss_fn(y_pred, y)
            # backprop to inputs of last layer
            self.model.backward(dyp)

            # attach grads to params (레이어들이 dW/db 보유 가정 -> param.grad에 연결)
            for (p, g, name) in self.model.parameters():
                if g is not None:
                    setattr(p, "grad", g)

            # (옵션) grad clip
            if grad_clip is not None:
                self._clip_grad_norm(grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

            total += float(loss); steps += 1
            for cb in self.cbs: cb.on_batch_end(epoch=epoch, step=step, loss=float(loss))
        if self.scheduler: self.scheduler.step()
        return total / max(steps,1)

    def fit(self, train_loader, epochs: int, grad_clip: Optional[float]=None):
        for ep in range(1, epochs+1):
            for cb in self.cbs: cb.on_epoch_start(epoch=ep)
            loss = self.train_epoch(train_loader, ep, grad_clip=grad_clip)
            for cb in self.cbs: cb.on_epoch_end(epoch=ep, train_loss=loss)

    def _clip_grad_norm(self, max_norm: float):
        import math
        sq = 0.0
        ps = []
        for (p, _, _) in self.model.parameters():
            g = getattr(p, "grad", None)
            if g is None: continue
            ps.append((p, g))
            sq += float((g*g).sum())  # ndarray-like
        norm = math.sqrt(sq) if sq>0 else 0.0
        if norm > max_norm and norm > 0:
            scale = max_norm / norm
            for p, g in ps:
                g[...] = g * scale
