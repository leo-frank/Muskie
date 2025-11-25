import torch
import math


def generate_connected_masks(
    B: int, V: int, H: int, W: int,
    ratio: float,
    *, device=None, dtype=torch.uint8,
    ar_range=(0.3, 3.0),
    mode: str = "rectangle"
) -> torch.Tensor:
    device = device or "cpu"

    mask = torch.zeros((B, V, H, W), device=device, dtype=dtype)
    r = ratio

    if mode == "rectangle":
        total = H * W
        for b in range(B):
            for v in range(V):

                target_area = max(1, int(round(r * total)))

                log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
                ar = math.exp(torch.empty(()).uniform_(log_min, log_max).item())  # ar = w/h

                h = max(1, int(round(math.sqrt(target_area / ar))))
                w = max(1, int(round(ar * h)))

                h = min(h, H)
                w = min(w, W)

                h = max(1, min(h, H))
                w = max(1, min(w, W))

                top = 0 if H == h else int(torch.randint(0, H - h + 1, (1,)).item())
                left = 0 if W == w else int(torch.randint(0, W - w + 1, (1,)).item())
                mask[b, v, top:top+h, left:left+w] = 1

    elif mode == "random_walk":
        for b in range(B):
            for v in range(V):

                target = max(1, int(round(r * H * W)))
                visited = torch.zeros((H, W), device=device, dtype=torch.bool)

                y = int(torch.randint(0, H, (1,)).item())
                x = int(torch.randint(0, W, (1,)).item())

                from collections import deque
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                filled = 0

                while q and filled < target:
                    cy, cx = q.popleft()
                    mask[b, v, cy, cx] = 1
                    filled += 1
                    if filled >= target:
                        break
                    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                    idx = torch.randperm(4)
                    for i in idx.tolist():
                        dy, dx = dirs[i]
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if filled < target:
                    ys, xs = torch.nonzero(mask[b, v] > 0, as_tuple=True)
                    border = []
                    for yy, xx in zip(ys.tolist(), xs.tolist()):
                        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                            ny, nx = yy + dy, xx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[b, v, ny, nx] == 0:
                                border.append((ny, nx))
                    if border:
                        border = list(set(border))
                        need = target - filled
                        take = min(need, len(border))
                        idxs = torch.randperm(len(border))[:take].tolist()
                        for k in idxs:
                            ny, nx = border[k]
                            mask[b, v, ny, nx] = 1
    elif mode == "ellipse":
        total = H * W
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij"
        )
        for b in range(B):
            for v in range(V):
                target_area = max(1, int(round(r * total)))
                base_r = math.sqrt(target_area / math.pi)
                ar = math.exp(torch.empty(()).uniform_(math.log(ar_range[0]), math.log(ar_range[1])).item())
                ry = max(1, int(round(base_r / math.sqrt(ar))))
                rx = max(1, int(round(base_r * math.sqrt(ar))))
                ry, rx = min(ry, H//2), min(rx, W//2)

                if H - 2*ry > 0:
                    cy = int(torch.randint(ry, H - ry, (1,)).item())
                else:
                    cy = H // 2
                if W - 2*rx > 0:
                    cx = int(torch.randint(rx, W - rx, (1,)).item())
                else:
                    cx = W // 2

                ellipse = ((yy - cy)**2 / (ry**2 + 1e-6) + (xx - cx)**2 / (rx**2 + 1e-6)) <= 1
                mask[b, v] = ellipse.to(dtype)
    elif mode == "blob":
        smooth_k = 10
        total = H * W
        target = max(1, int(round(r * total)))
        noise = torch.randn(B*V, 1, H, W, device=device)
        kernel_size = min(smooth_k, H, W)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = torch.nn.functional.avg_pool2d(noise, kernel_size, stride=1, padding=kernel_size//2)
        flat = smoothed.view(B*V, -1)
        kth = torch.topk(flat, target, dim=-1).values.min(dim=-1, keepdim=True).values
        mask = (flat >= kth).view(B, V, H, W)
    elif mode == "random":
        total = H * W
        target = max(1, int(round(r * total)))
        noise = torch.randn(B*V, 1, H, W, device=device)
        flat = noise.view(B*V, -1)
        kth = torch.topk(flat, target, dim=-1).values.min(dim=-1, keepdim=True).values
        mask = (flat >= kth).view(B, V, H, W)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mask.to(dtype)
