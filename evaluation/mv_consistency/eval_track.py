import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from evaluation.mv_consistency.utils import (
    build_loader,
    build_model,
    calculate_gt_tracks,
    compute_errors_for_sample,
    find_correspondences_feature_based,
    generate_dataset_report,
    get_backward_flow_from_attn,
    set_all_seeds,
    visualize_comparison_tracks,
)


def _infer_dataset_name(cfg: DictConfig) -> str:
    target = str(getattr(cfg.dataset, "_target_", "")).lower()
    if "navi" in target:
        return "navi"
    if "scannet" in target:
        return "scannet"
    return "dataset"


@hydra.main("./configs", "track", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    model = build_model(cfg).to("cuda")

    dataset_name = _infer_dataset_name(cfg)
    loader_kwargs = {}
    loader = build_loader(cfg.dataset, "valid", 1, 1, **loader_kwargs)

    dataset_2d_errors = []
    dataset_3d_errors = []
    model_name = (
        cfg.backbone.model_name
        if not hasattr(model, "checkpoint_name")
        else model.checkpoint_name
    )

    for idx, batch in enumerate(
        tqdm(loader, desc=f"Evaluating {model_name} on {dataset_name}")
    ):
        images = batch["image"][0].cuda(non_blocking=True)
        images_no_norm = images.clone()
        depths = batch["depth"][0].cuda(non_blocking=True)
        intrinsics = batch["intrinsics"][0].cuda(non_blocking=True)
        Rts = batch["Rt"][0].cuda(non_blocking=True)
        masks = batch["mask"][0].cuda(non_blocking=True)
        device = images.device

        if 'muskie' in model_name:
            mean=[0.5, 0.5, 0.5]; std=[0.5, 0.5, 0.5]
            images = (images - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / torch.tensor(std, device=device).view(1, 3, 1, 1)
        else:
            mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]
            images = (images - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / torch.tensor(std, device=device).view(1, 3, 1, 1)

        V, C, H, W = images.shape
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(V, 1, 1)

        num_tracks = cfg.num_corr
        masked_coords = masks[0].squeeze().nonzero(as_tuple=False)
        if len(masked_coords) == 0:
            print(f"Warning: No valid mask found for sample {idx}. Skipping.")
            continue
        if len(masked_coords) < num_tracks:
            print(
                f"Warning: Not enough points in mask for sample {idx}. Using {len(masked_coords)} points."
            )
            num_tracks = len(masked_coords)

        set_all_seeds(42 + idx)
        indices = torch.randperm(len(masked_coords))[:num_tracks]
        start_points = masked_coords[indices][:, [1, 0]].float()

        pred_tracks = torch.zeros((num_tracks, V, 2), device="cuda")
        pred_tracks[:, 0, :] = start_points

        with torch.no_grad():
            if "croco" in model_name:
                source_img_resized = F.interpolate(
                    images[0:1], size=(512, 512), mode="bilinear", align_corners=False
                )
                current_points = start_points
                for v_idx in range(1, V):
                    target_img_resized = F.interpolate(
                        images[v_idx : v_idx + 1],
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    )
                    norm_u = 2.0 * current_points[:, 0] / (W - 1) - 1.0
                    norm_v = 2.0 * current_points[:, 1] / (H - 1) - 1.0
                    grid = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
                    flow_s_to_t = model(
                        source_img_resized,
                        target_img_resized,
                        output_correlation="ca_map",
                    )
                    flow_s_to_t_upsampled = F.interpolate(
                        flow_s_to_t, size=(H, W), mode="bilinear", align_corners=False
                    )
                    flow_s_to_t_upsampled[:, 0, :, :] *= W / 640.0
                    flow_s_to_t_upsampled[:, 1, :, :] *= H / 480.0
                    flow_vectors = (
                        F.grid_sample(flow_s_to_t_upsampled, grid, align_corners=False)
                        .squeeze()
                        .T
                    )
                    if flow_vectors.dim() == 1:
                        flow_vectors = flow_vectors.unsqueeze(0)
                    next_points = current_points + flow_vectors
                    pred_tracks[:, v_idx, :] = next_points
            elif "muskie" in model_name:
                rgbs = images.unsqueeze(0)
                feats, attn_maps = model.get_latent_embeddings(rgbs, ret_attn=True)
                hp, wp = H // model.patch_size, W // model.patch_size
                for v_idx in range(1, V):
                    flow_0_to_v = get_backward_flow_from_attn(
                        v_idx, 0, attn_maps, hp, wp, model.patch_embed.patch_size, H, W
                    )
                    norm_u = 2.0 * start_points[:, 0] / (W - 1) - 1.0
                    norm_v = 2.0 * start_points[:, 1] / (H - 1) - 1.0
                    grid = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
                    flow_vectors = (
                        F.grid_sample(flow_0_to_v, grid, align_corners=False)
                        .squeeze()
                        .T
                    )
                    if flow_vectors.dim() == 1:
                        flow_vectors = flow_vectors.unsqueeze(0)
                    pred_tracks[:, v_idx, :] = start_points + flow_vectors
            else:
                rgbs = batch["image"].cuda()
                B, V, C, H, W = rgbs.shape
                rgbs = rgbs.view(B * V, C, H, W)
                feats = model(rgbs)
                Hf, Wf = feats.shape[-2:]
                feat_s = feats[0].unsqueeze(0)
                uv_s_feat_space = start_points * torch.tensor(
                    [Wf / W, Hf / H], device="cuda"
                )
                for v_idx in range(1, V):
                    feat_t = feats[v_idx].unsqueeze(0)
                    uv_t_feat_space = find_correspondences_feature_based(
                        uv_s_feat_space, feat_s, feat_t
                    )
                    pred_tracks[:, v_idx, :] = uv_t_feat_space * torch.tensor(
                        [W / Wf, H / Hf], device="cuda"
                    )

        gt_tracks = calculate_gt_tracks(
            start_points,
            depths,
            intrinsics,
            Rts,
            source_inverse=True,
            target_inverse=False,
            record_all_projections=False,
            check_start_in_bounds=True,
            invalidate_if_never_visible=True,
        )
        nrow = 4

        save_path = (
            f"viz/{dataset_name}_tracks_{num_tracks}/{model_name}/comparison_{idx}.png"
        )
        visualize_comparison_tracks(
            [img for img in images_no_norm], pred_tracks, gt_tracks, save_path, nrow=nrow
        )

        errors_2d_sample, errors_3d_sample = compute_errors_for_sample(
            pred_tracks, gt_tracks, depths, intrinsics
        )
        if errors_2d_sample.numel() > 0:
            dataset_2d_errors.append(errors_2d_sample.cpu())
            dataset_3d_errors.append(errors_3d_sample.cpu())

    generate_dataset_report(
        dataset_2d_errors,
        dataset_3d_errors,
        model_name,
        f"results/{dataset_name}_tracking",
    )


if __name__ == "__main__":
    main()
