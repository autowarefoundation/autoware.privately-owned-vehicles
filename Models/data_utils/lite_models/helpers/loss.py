import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, loss_cfg):
        #wrapper of normal segmentation losses, like the crossentropy loss
        super(SegmentationLoss, self).__init__()
        self.loss_cfg = loss_cfg

        self.loss_type = loss_cfg.get("type", "cross_entropy").lower()
        self.ignore_index = loss_cfg.get("ignore_index", 255)


        #build class weights if provided
        self.class_weights = loss_cfg.get("class_weights", None)

        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

        self.loss_fn = None

        #building loss function based on the configuration file
        if self.loss_type == "cross_entropy":
            #apply the cross entropy loss:
            """
            Cross Entropy Loss for multi-class segmentation.
            example : logits = tensors of shape [N,C,H,W]
                    targets = tensors of shape [H,W] with class indices
            """
            self.loss_fn =  nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            

    def forward(self, logits, targets):
        """
        logits: Tensor of shape [N, C, H, W] - raw output from the model
        targets: Tensor of shape [N, H, W] - ground truth class indices
        """
        return self.loss_fn(logits, targets)
    

class LanesLoss(nn.Module):
    """

        loss(channel) =
            BCEWithLogitsLoss(pred, gt)
          + MultiScaleEdgeLoss(pred, gt)

        total_loss =
            2 * left_lane_loss
          + 2 * right_lane_loss
          + 1 * other_lane_loss
    """

    def __init__(self, device="cuda", downsample_factor: int = 4):
        super().__init__()

        assert downsample_factor >= 1, "downsample_factor must be >= 1"
        assert downsample_factor & (downsample_factor - 1) == 0, \
            "downsample_factor must be a power of 2"

        self.downsample_factor = downsample_factor
        self.bce = nn.BCEWithLogitsLoss()


        # Sobel filters (exact values from original code)
        gx = torch.tensor([
            [ 0.125,  0.0,   -0.125],
            [ 0.25,   0.0,   -0.25 ],
            [ 0.125,  0.0,   -0.125],
        ], dtype=torch.float32).view(1, 1, 3, 3)

        gy = torch.tensor([
            [ 0.125,  0.25,   0.125],
            [ 0.0,    0.0,    0.0  ],
            [-0.125, -0.25,  -0.125],
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("gx_filter", gx)
        self.register_buffer("gy_filter", gy)

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.threshold = nn.Threshold(0.0, 1.0)


    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def forward(self, logits, gt):
        """
        logits: [B, 3, H, W]
        gt:     [B, 3, H, W]   (already downsampled as in original pipeline)
        """

        pred_left  = logits[:, 0, :, :]
        pred_right = logits[:, 1, :, :]
        pred_other = logits[:, 2, :, :]

        gt_left  = gt[:, 0, :, :]
        gt_right = gt[:, 1, :, :]
        gt_other = gt[:, 2, :, :]

        left_loss  = self._lane_loss(pred_left,  gt_left)
        right_loss = self._lane_loss(pred_right, gt_right)
        other_loss = self._lane_loss(pred_other, gt_other)

        total_loss = 2.0 * left_loss + 2.0 * right_loss + 1.0 * other_loss
        return total_loss


    # -------------------------------------------------
    # Lane loss = BCE + MultiScaleEdge
    # -------------------------------------------------
    def _lane_loss(self, pred, gt):

        if self.downsample_factor > 1:
            gt = self._downsample_gt(gt, factor=self.downsample_factor)

        seg_loss  = self.bce(pred, gt)
        edge_loss = self._multi_scale_edge_loss(pred, gt)
        return seg_loss + edge_loss


    def _downsample_gt(self, gt, factor=4):
        """
        gt: [B, H, W]
        returns: [B, H/factor, W/factor]
        """
        gt = gt.unsqueeze(1).float()  # [B,1,H,W]

        for _ in range(int(torch.log2(torch.tensor(factor)))):
            gt = F.max_pool2d(gt, kernel_size=2, stride=2)

        return gt.squeeze(1)  # [B,H_ds,W_ds]

    # -------------------------------------------------
    # Multi-scale edge loss
    # -------------------------------------------------
    def _multi_scale_edge_loss(self, pred, gt):

        # ReLU + threshold as in original code
        pred = self.relu(pred)
        pred = self.threshold(pred)

        pred_d1 = self.avg_pool(pred)
        pred_d2 = self.avg_pool(pred_d1)
        pred_d3 = self.avg_pool(pred_d2)
        pred_d4 = self.avg_pool(pred_d3)

        gt_d1 = self.avg_pool(gt)
        gt_d2 = self.avg_pool(gt_d1)
        gt_d3 = self.avg_pool(gt_d2)
        gt_d4 = self.avg_pool(gt_d3)

        edge_loss_d0 = self._edge_loss(pred,     gt)
        edge_loss_d1 = self._edge_loss(pred_d1,  gt_d1)
        edge_loss_d2 = self._edge_loss(pred_d2,  gt_d2)
        edge_loss_d3 = self._edge_loss(pred_d3,  gt_d3)
        edge_loss_d4 = self._edge_loss(pred_d4,  gt_d4)

        return (
            edge_loss_d0 +
            edge_loss_d1 +
            edge_loss_d2 +
            edge_loss_d3 +
            edge_loss_d4
        ) / 5.0


    # -------------------------------------------------
    # Edge loss (Sobel filters + L1)
    # -------------------------------------------------
    def _edge_loss(self, pred, gt):
        # pred, gt: [B, H, W] → add channel dim
        pred = pred.unsqueeze(1)
        gt   = gt.unsqueeze(1)

        Gx_pred = F.conv2d(pred, self.gx_filter, padding=1)
        Gy_pred = F.conv2d(pred, self.gy_filter, padding=1)

        Gx_gt = F.conv2d(gt, self.gx_filter, padding=1)
        Gy_gt = F.conv2d(gt, self.gy_filter, padding=1)

        edge_diff = torch.abs(Gx_pred - Gx_gt) + torch.abs(Gy_pred - Gy_gt)
        return torch.mean(edge_diff)


# ============================================================
# DEPTH Loss
# ============================================================

EDGE_SCALE_FACTOR_DEFAULT = 4.0  # same as self.edge_scale_factor in Scene3DTrainer

class DepthLoss(nn.Module):
    """
    Exact reproduction of Scene3DTrainer loss:
      - SSI normalization
      - Robust mAE (90% quantile)
      - Multi-scale edge loss (scales 0..N-1)
      - total = mAE + edge_factor * edge_loss
    """

    def __init__(self, loss_cfg=None, is_train=True):
        super().__init__()

        self.loss_cfg = loss_cfg or {}
        self.edge_scale_factor = self.loss_cfg.get("edge_factor", EDGE_SCALE_FACTOR_DEFAULT)
        self.num_scales = self.loss_cfg.get("num_scales", 4)

        assert self.num_scales >= 1

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build filters and register buffers
        gx, gy = self.build_grad_filters(device)
        self.register_buffer("gx_filter", gx)
        self.register_buffer("gy_filter", gy)

        #distinguish between train and val (for absrel calculation)
        self.is_train = is_train

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, pred, gt):
        pred = pred.float()
        gt   = gt.float()

        # SSI normalization
        pred_ssi = self.get_ssi_norm_tensor(pred)
        gt_ssi   = self.get_ssi_norm_tensor(gt)

        # mAE robust loss, averaged across batch
        mAE_loss = self.calc_mAE_ssi_loss_robust(pred_ssi, gt_ssi)

        # Multi-scale edge loss
        edge_loss = self.calc_multi_scale_ssi_edge_loss(pred_ssi, gt_ssi)

        total = mAE_loss + self.edge_scale_factor * edge_loss


        #only return total, mae and edge during training
        if self.is_train:
            return total, mAE_loss, edge_loss
        
        metrics = self.calc_metrics(pred, gt)
        return total, mAE_loss, edge_loss, metrics



    # ------------------------------------------------------------
    # SSI Normalization
    # ------------------------------------------------------------
    def get_ssi_norm_tensor(self, t):
        """
        t: [B,1,H,W]
        SSI normalization per image
        """
        if t.ndim == 3:
            t = t.unsqueeze(1)

        B, C, H, W = t.shape
        t_flat = t.view(B, -1)

        t_min  = t_flat.min(dim=1, keepdim=True)[0]
        t_mean = t_flat.mean(dim=1, keepdim=True)
        t_max  = t_flat.max(dim=1, keepdim=True)[0]

        denom = (t_max - t_mean).clamp(min=1e-6)
        t_norm = (t_flat - t_min) / denom

        return t_norm.view(B, C, H, W)


    # ------------------------------------------------------------
    # Robust mAE (90th percentile mask)
    # ------------------------------------------------------------
    def calc_mAE_ssi_loss_robust(self, pred_ssi, gt_ssi):
        diff = torch.abs(pred_ssi - gt_ssi)
        diff_flat = diff.reshape(-1)

        q = torch.quantile(diff_flat, 0.9, interpolation="linear")
        mask = diff_flat < q

        if mask.sum() == 0:
            return diff.mean()

        return diff_flat[mask].mean()


    # ------------------------------------------------------------
    # Gradient filters
    # ------------------------------------------------------------
    def build_grad_filters(self, device):
        gx = torch.tensor(
            [[0.125, 0.0, -0.125],
             [0.25,  0.0, -0.25 ],
             [0.125, 0.0, -0.125]],
            dtype=torch.float32, device=device
        ).view(1,1,3,3)

        gy = torch.tensor(
            [[0.125, 0.25,  0.125],
             [0.0,   0.0,   0.0  ],
             [-0.125, -0.25, -0.125]],
            dtype=torch.float32, device=device
        ).view(1,1,3,3)

        return gx, gy


    # ------------------------------------------------------------
    # Single-scale edge loss
    # ------------------------------------------------------------
    def calc_edge_ssi_loss(self, pred_ssi, gt_ssi):
        Gx_pred = F.conv2d(pred_ssi, self.gx_filter, padding=1)
        Gy_pred = F.conv2d(pred_ssi, self.gy_filter, padding=1)

        Gx_gt = F.conv2d(gt_ssi, self.gx_filter, padding=1)
        Gy_gt = F.conv2d(gt_ssi, self.gy_filter, padding=1)

        diff = torch.abs(Gx_pred - Gx_gt) + torch.abs(Gy_pred - Gy_gt)
        return diff.mean()


    # ------------------------------------------------------------
    # Multi-scale edge loss (true implementation)
    # ------------------------------------------------------------
    def calc_multi_scale_ssi_edge_loss(self, pred_ssi, gt_ssi):

        edge_losses = []

        p = pred_ssi
        g = gt_ssi

        for _ in range(self.num_scales):

            # Compute edge loss at current resolution
            edge_losses.append(self.calc_edge_ssi_loss(p, g))

            # Downsample for next scale
            p = F.avg_pool2d(p, 2, stride=2)
            g = F.avg_pool2d(g, 2, stride=2)

        return torch.mean(torch.stack(edge_losses))



    def calc_absrel(self, pred, gt, eps=1e-6):
        """
        pred, gt: torch tensors [B,1,H,W] or [H,W]
        NOTE: scale alignment is applied before computing absrel (otherwise it is meaningless)
        """

        #remove invalid pixels (below eps)
        valid = gt > eps
        pred = pred * valid
        gt   = gt * valid

        #align the prediction to the ground truth
        pred_aligned = self.align_scale_shift(pred, gt)

        #calculate absrel
        diff = torch.abs(pred_aligned - gt)
        absrel = diff[valid] / (gt[valid] + eps)

        return absrel.mean()


    def align_scale_shift(self, pred, gt, eps=1e-6):
        """
        pred, gt: [B,1,H,W] or [B,H,W]
        returns: pred aligned to gt
        """
        B = pred.shape[0]

        pred_flat = pred.view(B, -1)
        gt_flat   = gt.view(B, -1)

        ones = torch.ones_like(pred_flat)
        A = torch.stack([pred_flat, ones], dim=2)  # [B, N, 2]

        # Least squares per batch element
        ATA = torch.matmul(A.transpose(1, 2), A)                # [B,2,2]
        ATy = torch.matmul(A.transpose(1, 2), gt_flat.unsqueeze(-1))  # [B,2,1]

        eye = torch.eye(2, device=pred.device).unsqueeze(0)
        x = torch.linalg.solve(ATA + eps * eye, ATy)             # [B,2,1]

        a = x[:, 0, 0]   # [B]
        b = x[:, 1, 0]   # [B]

        pred_aligned = a[:, None] * pred_flat + b[:, None]      # [B, N]

        return pred_aligned.view_as(pred)



    def calc_metrics(
        self,
        pred,
        gt,
        min_depth=1e-3,
        max_depth=None,
        eps=1e-6,
    ):
        """
        Monodepth2-style evaluation metrics.
        pred, gt: [B,1,H,W]
        """

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)

        B = pred.shape[0]

        absrels = []
        sqrels = []
        rmses = []
        rmses_log = []
        a1s = []
        a2s = []
        a3s = []

        for b in range(B):
            p = pred[b, 0]
            g = gt[b, 0]

            # -------------------------
            # VALID MASK
            # -------------------------
            valid = g > min_depth
            if max_depth is not None:
                valid = valid & (g < max_depth)

            if valid.sum() < 50:
                continue

            p = p[valid]
            g = g[valid]

            # -------------------------
            # MEDIAN SCALING (NO SHIFT)
            # -------------------------
            scale = torch.median(g) / (torch.median(p) + eps)
            p = p * scale

            # -------------------------
            # CLAMP
            # -------------------------
            if max_depth is not None:
                p = torch.clamp(p, min_depth, max_depth)
            else:
                p = torch.clamp(p, min_depth)

            # -------------------------
            # METRICS
            # -------------------------
            diff = torch.abs(g - p)

            abs_rel = torch.mean(diff / g)
            sq_rel = torch.mean(diff ** 2 / g)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            rmse_log = torch.sqrt(torch.mean((torch.log(g) - torch.log(p)) ** 2))

            thresh = torch.max(g / p, p / g)
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25 ** 2).float().mean()
            a3 = (thresh < 1.25 ** 3).float().mean()

            absrels.append(abs_rel)
            sqrels.append(sq_rel)
            rmses.append(rmse)
            rmses_log.append(rmse_log)
            a1s.append(a1)
            a2s.append(a2)
            a3s.append(a3)

        return {
            "abs_rel": torch.stack(absrels).mean(),
            "sq_rel": torch.stack(sqrels).mean(),
            "rmse": torch.stack(rmses).mean(),
            "rmse_log": torch.stack(rmses_log).mean(),
            "a1": torch.stack(a1s).mean(),
            "a2": torch.stack(a2s).mean(),
            "a3": torch.stack(a3s).mean(),
        }
