"""
PFM-VEPAR visualization: per-attribute accuracy chart + sample predictions.
Usage: python visualize.py --cfg ./configs/pedes_baseline/EventPAR.yaml
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from dataset.augmentation import get_transform
from models.base_block import FeatClassifier
from models.model_factory import build_backbone, build_classifier
from models.backbone import vit
from tools.utils import set_seed, str2bool

set_seed(605)


def get_attribute_names(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    return list(data.attr_name)


def build_model(cfg, ckpt_path, attr_num):
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    if cfg.CROSS_ATTENTION.ENABLE and cfg.CROSS_ATTENTION.FUSION_TYPE == 'adaptive':
        classifier_input_dim = c_output
    else:
        classifier_input_dim = c_output * 2

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=attr_num,
        c_in=classifier_input_dim,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE,
    )

    model = FeatClassifier(
        backbone, classifier,
        bn_wd=cfg.TRAIN.BN_WD,
        enable_dct=cfg.DCT.ENABLE,
        dct_on_rgb=cfg.DCT.APPLY_TO_RGB,
        dct_on_event=cfg.DCT.APPLY_TO_EVENT,
        enable_stem=cfg.STEM.ENABLE,
        stem_out_chans=cfg.STEM.OUT_CHANS,
        enable_cross_attention=cfg.CROSS_ATTENTION.ENABLE,
        cross_attn_layers=cfg.CROSS_ATTENTION.NUM_LAYERS,
        cross_attn_heads=cfg.CROSS_ATTENTION.NUM_HEADS,
        fusion_type=cfg.CROSS_ATTENTION.FUSION_TYPE,
        enable_hopfield=cfg.HOPFIELD.ENABLE,
        hopfield_apply_to_rgb=cfg.HOPFIELD.APPLY_TO_RGB,
        hopfield_apply_to_event=cfg.HOPFIELD.APPLY_TO_EVENT,
        hopfield_n_prototype=cfg.HOPFIELD.N_PROTOTYPE,
        hopfield_dropout=cfg.HOPFIELD.DROPOUT,
        cfg=cfg,
    )
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Prefer EMA weights; add 'module.' prefix if needed (DataParallel)
    sd = ckpt.get('state_dict_ema', ckpt.get('state_dicts', ckpt))
    if not list(sd.keys())[0].startswith('module.'):
        sd = {f'module.{k}': v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    print(f"Loaded checkpoint from {ckpt_path}")
    if 'metric' in ckpt:
        print(f"Checkpoint mA: {ckpt['metric']}")
    model.eval()
    return model


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def draw_predictions(rgb_img, gt_label, pred_probs, attr_names, threshold=0.5):
    draw = ImageDraw.Draw(rgb_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    pos_indices = np.where(gt_label == 1)[0]
    lines = []
    for idx in pos_indices[:9]:
        attr = attr_names[idx]
        gt = int(gt_label[idx])
        pred = 1 if pred_probs[idx] > threshold else 0
        prob = pred_probs[idx]
        status = "TP" if (gt == 1 and pred == 1) else ("FN" if (gt == 1 and pred == 0) else "N/A")
        lines.append(f"[{status}] {attr}: GT=1 Pred={pred} ({prob:.2f})")

    text_h = len(lines) * 14 + 8
    overlay = Image.new('RGBA', (rgb_img.width, text_h), (0, 0, 0, 180))
    rgb_img = rgb_img.convert('RGBA')
    rgb_img.paste(overlay, (0, rgb_img.height - text_h), overlay)

    y = rgb_img.height - text_h + 4
    for line in lines:
        draw.text((5, y), line, fill=(255, 255, 255, 255), font=font)
        y += 14
    return rgb_img.convert('RGB')


def main(cfg, args):
    attr_names = get_attribute_names(cfg.DATASET.PKL)
    print(f"Attributes: {len(attr_names)}")

    _, valid_tsfm = get_transform(cfg)
    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)

    # Non-shuffled loader for full evaluation
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Test set: {len(valid_set)} samples')

    model = build_model(cfg, args.checkpoint, len(attr_names))
    os.makedirs(args.output_dir, exist_ok=True)

    all_gt, all_probs, all_img_paths = [], [], []
    saved = 0

    with torch.no_grad():
        for rgb_imgs, event_imgs, gt_label, imgname in tqdm(valid_loader, desc="Inference"):
            rgb_imgs = rgb_imgs.cuda()
            event_imgs = event_imgs.cuda()
            logits, _ = model(rgb_imgs, event_imgs)
            probs = torch.sigmoid(logits[0]).cpu().numpy()
            gt_np = gt_label.cpu().numpy()

            all_gt.append(gt_np)
            all_probs.append(probs)

            # Save a few sample visualizations (from first batches)
            for i in range(len(gt_np)):
                if saved >= args.num_samples:
                    break
                img_t = rgb_imgs[i].cpu().squeeze(0)
                rgb_img = denormalize(img_t)
                result = draw_predictions(rgb_img, gt_np[i], probs[i], attr_names)
                result.save(os.path.join(args.output_dir, f'sample_{saved:04d}_{imgname[i]}.jpg'))
                saved += 1

    all_gt = np.concatenate(all_gt, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    print(f"Total processed: {len(all_gt)} samples")

    # ---- Per-attribute accuracy chart ----
    per_attr_acc = []
    for j in range(len(attr_names)):
        preds = (all_probs[:, j] > 0.5).astype(int)
        acc = (preds == all_gt[:, j]).mean()
        per_attr_acc.append(acc)

    mA = np.mean(per_attr_acc)

    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ['#2ecc71' if a > 0.85 else '#f39c12' if a > 0.7 else '#e74c3c' for a in per_attr_acc]
    ax.bar(range(len(attr_names)), per_attr_acc, color=colors)
    ax.set_xticks(range(len(attr_names)))
    ax.set_xticklabels(attr_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'PFM-VEPAR Per-Attribute Accuracy on EventPAR (mA = {mA:.4f})')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=mA, color='blue', linestyle='-', alpha=0.7, label=f'mA = {mA:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'per_attribute_accuracy.png'), dpi=150)
    print(f"Chart saved to {args.output_dir}/per_attribute_accuracy.png")

    # ---- Overall metrics ----
    from metrics.pedestrian_metrics import get_pedestrian_metrics
    valid_result = get_pedestrian_metrics(all_gt, all_probs)
    print(f"\n--- Full Test Set Results (10000 samples) ---")
    print(f"mA:        {valid_result.ma:.4f}")
    print(f"Acc:       {valid_result.instance_acc:.4f}")
    print(f"Prec:      {valid_result.instance_prec:.4f}")
    print(f"Recall:    {valid_result.instance_recall:.4f}")
    print(f"F1:        {valid_result.instance_f1:.4f}")
    print(f"Pos_recall: {np.mean(valid_result.label_pos_recall):.4f}")
    print(f"Neg_recall: {np.mean(valid_result.label_neg_recall):.4f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PFM-VEPAR Visualization")
    parser.add_argument("--cfg", type=str, default="./configs/pedes_baseline/EventPAR.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="./logs/EventPAR/default/img_model/ckpt_max_2026-05-25_10:57:58.pth")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="./visualization_results")
    args = parser.parse_args()
    update_config(cfg, args)
    main(cfg, args)
