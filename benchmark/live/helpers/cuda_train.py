"""An ordinary CUDA training script with no reflop in it. Run as `reflop run python cuda_train.py`."""

import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "needs CUDA (reflop run provides it)"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
dev = "cuda"
amp = "--amp" in sys.argv
model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                      nn.Flatten(), nn.Linear(256, 10)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scaler = torch.amp.GradScaler("cuda", enabled=amp)
g = torch.Generator().manual_seed(1)
X = torch.randn(512, 3, 16, 16, generator=g)
Y = (X.mean((1, 2, 3)) > 0).long()
losses = []
for epoch in range(3):
    tot = 0.0
    for i in range(0, 512, 64):
        x, y = X[i:i + 64].to(dev), Y[i:i + 64].to(device=dev)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item()
    losses.append(tot / 8)
p = next(model.parameters())
print(json.dumps({"device": str(p.device), "type": type(p.data).__name__, "losses": losses, "amp": amp}))
