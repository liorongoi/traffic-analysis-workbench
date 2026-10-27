"""YOLOX in PyTorch, for training only (LRN-02 vehicle detector adapter).

The layers and parameter names follow the YOLOX repository (Apache-2.0, Megvii), so its released
checkpoints (yolox_s.pth and others) load directly and a trained model exports to the same ONNX
layout the app already runs: 85 values per anchor = raw box offsets, objectness and class
probabilities (sigmoid), boxes decoded by the app with its grid (see detect.py).

The loss is YOLOX's: SimOTA assignment, IoU loss on boxes, binary cross-entropy on objectness and
classes. One addition for this app: a per-anchor `neg_weight` so areas nobody reviewed (partial
reviews, excluded regions, ignore boxes) never teach "background".

Imported only by the training command, so the app itself never needs PyTorch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {"s": (0.33, 0.50), "m": (0.67, 0.75), "l": (1.0, 1.0), "tiny": (0.33, 0.375), "nano": (0.33, 0.25)}


class BaseConv(nn.Module):
    def __init__(self, cin, cout, k, s, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, (k - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, shortcut=True, expansion=0.5):
        super().__init__()
        h = int(cout * expansion)
        self.conv1 = BaseConv(cin, h, 1, 1)
        self.conv2 = BaseConv(h, cout, 3, 1)
        self.use_add = shortcut and cin == cout

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        h = int(cout * expansion)
        self.conv1 = BaseConv(cin, h, 1, 1)
        self.conv2 = BaseConv(cin, h, 1, 1)
        self.conv3 = BaseConv(2 * h, cout, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(h, h, shortcut, 1.0) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class Focus(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = BaseConv(cin * 4, cout, k, 1)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, ks=(5, 9, 13)):
        super().__init__()
        h = cin // 2
        self.conv1 = BaseConv(cin, h, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in ks])
        self.conv2 = BaseConv(h * (len(ks) + 1), cout, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, dep, wid):
        super().__init__()
        c, d = int(wid * 64), max(round(dep * 3), 1)
        self.stem = Focus(3, c, 3)
        self.dark2 = nn.Sequential(BaseConv(c, c * 2, 3, 2), CSPLayer(c * 2, c * 2, d))
        self.dark3 = nn.Sequential(BaseConv(c * 2, c * 4, 3, 2), CSPLayer(c * 4, c * 4, d * 3))
        self.dark4 = nn.Sequential(BaseConv(c * 4, c * 8, 3, 2), CSPLayer(c * 8, c * 8, d * 3))
        self.dark5 = nn.Sequential(BaseConv(c * 8, c * 16, 3, 2), SPPBottleneck(c * 16, c * 16),
                                   CSPLayer(c * 16, c * 16, d, shortcut=False))

    def forward(self, x):
        x = self.dark2(self.stem(x))
        d3 = self.dark3(x)
        d4 = self.dark4(d3)
        return d3, d4, self.dark5(d4)


class YOLOPAFPN(nn.Module):
    def __init__(self, dep, wid):
        super().__init__()
        self.backbone = CSPDarknet(dep, wid)
        c3, c4, c5 = int(256 * wid), int(512 * wid), int(1024 * wid)
        n = round(3 * dep)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(c5, c4, 1, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, n, False)
        self.reduce_conv1 = BaseConv(c4, c3, 1, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, n, False)
        self.bu_conv2 = BaseConv(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, n, False)
        self.bu_conv1 = BaseConv(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, n, False)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn0 = self.lateral_conv0(x0)
        f0 = self.C3_p4(torch.cat((self.upsample(fpn0), x1), 1))
        fpn1 = self.reduce_conv1(f0)
        p2 = self.C3_p3(torch.cat((self.upsample(fpn1), x2), 1))
        p1 = self.C3_n3(torch.cat((self.bu_conv2(p2), fpn1), 1))
        p0 = self.C3_n4(torch.cat((self.bu_conv1(p1), fpn0), 1))
        return p2, p1, p0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, wid, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes, self.strides = num_classes, strides
        hid = int(256 * wid)
        self.stems, self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.cls_preds, self.reg_preds, self.obj_preds = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for cin in (256, 512, 1024):
            self.stems.append(BaseConv(int(cin * wid), hid, 1, 1))
            self.cls_convs.append(nn.Sequential(BaseConv(hid, hid, 3, 1), BaseConv(hid, hid, 3, 1)))
            self.reg_convs.append(nn.Sequential(BaseConv(hid, hid, 3, 1), BaseConv(hid, hid, 3, 1)))
            self.cls_preds.append(nn.Conv2d(hid, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hid, 4, 1))
            self.obj_preds.append(nn.Conv2d(hid, 1, 1))
        prior = -math.log((1 - 0.01) / 0.01)
        for conv in list(self.cls_preds) + list(self.obj_preds):
            nn.init.constant_(conv.bias, prior)

    def forward(self, feats):
        """Raw outputs (N, A, 5 + classes): box offsets, objectness logit, class logits; and the grid."""
        outs, grids, strides = [], [], []
        for k, x in enumerate(feats):
            x = self.stems[k](x)
            cf, rf = self.cls_convs[k](x), self.reg_convs[k](x)
            o = torch.cat((self.reg_preds[k](rf), self.obj_preds[k](rf), self.cls_preds[k](cf)), 1)
            n, c, h, w = o.shape
            outs.append(o.flatten(2).permute(0, 2, 1))
            yv, xv = torch.meshgrid(torch.arange(h, device=o.device), torch.arange(w, device=o.device), indexing="ij")
            grids.append(torch.stack((xv, yv), 2).reshape(-1, 2).float())
            strides.append(torch.full((h * w,), float(self.strides[k]), device=o.device))
        return torch.cat(outs, 1), torch.cat(grids, 0), torch.cat(strides, 0)


class YOLOX(nn.Module):
    def __init__(self, size: str = "s", num_classes: int = 80):
        super().__init__()
        dep, wid = SIZES[size]
        self.backbone = YOLOPAFPN(dep, wid)
        self.head = YOLOXHead(num_classes, wid)

    def forward(self, x):
        return self.head(self.backbone(x))


class ExportWrapper(nn.Module):
    """The ONNX layout the app runs: raw offsets, sigmoid objectness, sigmoid class scores."""

    def __init__(self, model: YOLOX):
        super().__init__()
        self.model = model

    def forward(self, x):
        raw, _, _ = self.model(x)
        return torch.cat((raw[..., :4], raw[..., 4:].sigmoid()), -1)


def load_checkpoint(model: YOLOX, path: str) -> list[str]:
    """Load a YOLOX repository checkpoint; returns the names that did not fit (empty when it matches)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    own = model.state_dict()
    fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(fit, strict=False)
    return sorted(set(own) - set(fit))


# ------------------------------------------------------------------ loss

def decode(raw, grid, stride):
    xy = (raw[..., :2] + grid) * stride[:, None]
    wh = torch.exp(raw[..., 2:4].clamp(max=10)) * stride[:, None]
    return torch.cat((xy, wh), -1)                     # cx, cy, w, h


def box_iou_cxcywh(a, b):
    """IoU between every box in a (n, 4) and b (m, 4), both cx, cy, w, h."""
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0).prod(-1)
    return inter / (a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter + 1e-9)


def iou_pairs(a, b):
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0).prod(-1)
    return inter / (a[:, 2:].prod(-1) + b[:, 2:].prod(-1) - inter + 1e-9)


@torch.no_grad()
def simota(boxes, obj, cls, gt_boxes, gt_cls, grid, stride, num_classes, center_radius=2.5):
    """Which anchors learn which ground-truth box (YOLOX's SimOTA). Returns (fg mask, gt index per fg anchor)."""
    A = boxes.shape[0]
    centers = (grid + 0.5) * stride[:, None]
    g1, g2 = gt_boxes[:, None, :2] - gt_boxes[:, None, 2:] / 2, gt_boxes[:, None, :2] + gt_boxes[:, None, 2:] / 2
    in_box = ((centers[None] > g1) & (centers[None] < g2)).all(-1)                       # (G, A)
    r = center_radius * stride[None, :, None]
    in_ctr = ((centers[None] > gt_boxes[:, None, :2] - r) & (centers[None] < gt_boxes[:, None, :2] + r)).all(-1)
    cand = (in_box | in_ctr).any(0)
    fg = torch.zeros(A, dtype=torch.bool, device=boxes.device)
    match = torch.zeros(A, dtype=torch.long, device=boxes.device)
    if not cand.any():
        return fg, match[fg]
    both = (in_box & in_ctr)[:, cand]
    ious = box_iou_cxcywh(gt_boxes, boxes[cand])                                       # (G, C)
    iou_cost = -torch.log(ious + 1e-8)
    p = (cls[cand].float().sigmoid() * obj[cand, None].float().sigmoid()).sqrt()
    onehot = F.one_hot(gt_cls, num_classes).float()
    cls_cost = F.binary_cross_entropy(p[None].expand(len(gt_cls), -1, -1).clamp(1e-6, 1 - 1e-6),
                                      onehot[:, None].expand(-1, p.shape[0], -1), reduction="none").sum(-1)
    cost = cls_cost + 3.0 * iou_cost + 1e5 * (~both)
    k = min(10, ious.shape[1])
    dyn = torch.topk(ious, k, 1).values.sum(1).int().clamp(min=1)
    matching = torch.zeros_like(cost)
    for g in range(len(gt_cls)):
        idx = torch.topk(cost[g], int(min(dyn[g], cost.shape[1])), largest=False).indices
        matching[g, idx] = 1.0
    multi = matching.sum(0) > 1
    if multi.any():
        best = cost[:, multi].argmin(0)
        matching[:, multi] = 0.0
        matching[best, multi.nonzero().squeeze(1)] = 1.0
    sel = matching.sum(0) > 0
    cand_idx = cand.nonzero().squeeze(1)
    fg[cand_idx[sel]] = True
    match[cand_idx[sel]] = matching[:, sel].argmax(0)
    return fg, match[fg]


def yolox_loss(raw, grid, stride, targets, neg_weight, num_classes):
    """targets: list per image of (G, 5) tensors [cls, cx, cy, w, h] in input pixels.
    neg_weight: (N, A) — 1 where an anchor may learn "no vehicle here", 0 where nobody reviewed."""
    boxes = decode(raw, grid, stride)
    obj_t = torch.zeros(raw.shape[:2], device=raw.device)
    obj_w = neg_weight.clone().float()
    iou_l, cls_l, nfg = raw.new_zeros(()), raw.new_zeros(()), 0
    for i, t in enumerate(targets):
        if len(t) == 0:
            continue
        gcls, gbox = t[:, 0].long(), t[:, 1:5]
        fg, m = simota(boxes[i], raw[i, :, 4], raw[i, :, 5:], gbox, gcls, grid, stride, num_classes)
        if not fg.any():
            continue
        nfg += int(fg.sum())
        ious = iou_pairs(boxes[i][fg], gbox[m])
        iou_l = iou_l + (1 - ious ** 2).sum()
        cls_t = F.one_hot(gcls[m], num_classes).float() * ious.detach()[:, None]
        cls_l = cls_l + F.binary_cross_entropy_with_logits(raw[i, fg, 5:], cls_t, reduction="sum")
        obj_t[i, fg] = 1.0
        obj_w[i, fg] = 1.0
    obj_l = (F.binary_cross_entropy_with_logits(raw[..., 4], obj_t, reduction="none") * obj_w).sum()
    n = max(nfg, 1)
    total = (5.0 * iou_l + obj_l + cls_l) / n
    return total, {"iou": float(iou_l.detach()) / n, "obj": float(obj_l.detach()) / n, "cls": float(cls_l.detach()) / n,
                   "fg": nfg}
