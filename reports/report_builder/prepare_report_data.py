"""Collect every number and data-driven sentence the report needs into one JSON file.

    python prepare_report_data.py <results_dir> <out.json> [--previous <results_dir_of_run_1>]
                                  [--manual manual_text.json] [--app-fig-dir DIR] [--static-fig-dir DIR]

<results_dir> is the extracted results.zip (folders figures/, metrics/, model/, sample_images/).
--previous     results of an earlier training run, used for the "iterative improvement" table.
--manual       interpretations written after looking at image figures (errors, Grad-CAM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD_SOURCE_NAMES = {"Garbage Classification v2 (Suman, 2024)": "Garbage Dataset v2 (Kunwar, 2026)"}
SOURCE_SHORT = {"TrashNet (Thung & Yang, 2016)": "TrashNet (Thung & Yang, 2016)",
                "Garbage Classification 12 classes (Mohamed, 2021)": "the 12-class Garbage Classification data set (Mohamed, 2021)",
                "Garbage Dataset v2 (Kunwar, 2026)": "the Garbage Dataset (Kunwar, 2026)"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def fmt(n):
    return f"{int(round(n)):,}"


def join_and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------------------------------------
# Data analysis (Section 1)
# ------------------------------------------------------------------------------------------
def dataset_part(R, m, summary):
    n = summary["n_images"]
    counts = pd.Series(summary["class_counts"]).reindex(CLASSES)
    R.update(
        n_images=n, n_images_fmt=fmt(n),
        exact_dup_files_fmt=fmt(summary["exact_duplicate_files"]),
        exact_dup_pct=pct(summary["exact_duplicate_files"] / n),
        exact_dup_groups_fmt=fmt(summary["exact_duplicate_groups"]),
        near_dup_images_fmt=fmt(summary["images_with_near_duplicate"]["<=4"]),
        near_dup_pct=pct(summary["images_with_near_duplicate"]["<=4"] / n),
        images_phash0_fmt=fmt(summary["images_with_near_duplicate"]["<=0"]),
        cross_pairs_4=summary["cross_class_pairs"]["<=4"],
        cross_pairs_0=summary["cross_class_pairs"]["<=0"],
        removed_exact_fmt=fmt(summary.get("removed_exact_duplicates", 0)),
        removed_conflicts_fmt=fmt(summary.get("removed_label_conflicts", 0)),
        dup_groups_multi_fmt=fmt(summary.get("n_dup_groups_multi", 0)),
        mean_class_fmt=fmt(round(counts.mean())), min_class_fmt=fmt(counts.min()),
        imbalance_ratio=f"{summary['imbalance_ratio_max_min']:.2f}",
        share_min=pct(summary["class_share_min_max"][0]), share_max=pct(summary["class_share_min_max"][1]),
        evenness=f"{summary['shannon_evenness']:.3f}",
        total_mb=f"{summary['total_size_mb']:.1f}",
        median_kb=f"{summary['file_size_kb']['50%']:.1f}",
        size_range=f"{summary['file_size_kb']['min']:.1f}–{summary['file_size_kb']['max']:.1f} KB",
        class_counts={k: int(v) for k, v in counts.items()},
    )

    stats = pd.read_csv(m / "image_stats.csv")
    stats = stats[stats["readable"].fillna(False).astype(bool)]
    # ---- exact duplicates per class (unique photos = distinct MD5 checksums)
    g = stats.groupby("class_name").agg(files=("md5", "size"), unique=("md5", "nunique")).reindex(CLASSES)
    g["copies"] = g["files"] - g["unique"]
    g["share"] = g["copies"] / g["files"]
    padded = g.drop(index=["glass", "trash"])
    md5_classes = stats.groupby("md5")["class_name"].nunique()
    group_sizes = stats.groupby("md5").size()
    R["dup"] = {
        "unique": g["unique"].astype(int).to_dict(), "copies": g["copies"].astype(int).to_dict(),
        "share": g["share"].round(4).to_dict(),
        "min_unique": int(g["unique"].min()), "min_unique_fmt": fmt(g["unique"].min()), "min_class": g["unique"].idxmin(),
        "max_unique": int(g["unique"].max()), "max_unique_fmt": fmt(g["unique"].max()), "max_class": g["unique"].idxmax(),
        "ratio_unique": float(g["unique"].max() / g["unique"].min()),
        "copy_share_range": f"{100 * padded['share'].min():.0f}–{100 * padded['share'].max():.0f}%",
        "glass_share": pct(g.loc["glass", "share"], 0),
        "unique_total": int(stats["md5"].nunique()),
        "all_pairs": bool(group_sizes[group_sizes > 1].eq(2).all()),
    }
    R["cross_md5_groups"] = int((md5_classes > 1).sum())
    R["cross_label_text"] = (
        f"Moreover, {R['cross_md5_groups']} identical pairs carry *different* labels – the same photo appears in two "
        "classes, a form of label noise."
        if R["cross_md5_groups"] else "No identical pair carries conflicting labels.")

    # ---- white backgrounds and correlations (Section 1.2)
    R["white_bg"] = (stats["white_background"] >= 0.6).groupby(stats["class_name"]).mean().reindex(CLASSES).round(4).to_dict()
    feats = ["file_size_kb", "mean_r", "mean_g", "mean_b", "brightness", "contrast", "saturation", "colourfulness",
             "sharpness", "entropy", "edge_density", "white_background"]
    corr = stats[feats].corr(method="spearman")
    R["corr"] = {f"{a}__{b}": round(float(corr.loc[a, b]), 3) for a in feats for b in feats if a != b}
    med = {}
    for feat in ["brightness", "contrast", "saturation", "colourfulness", "sharpness", "entropy", "edge_density",
                 "white_background", "file_size_kb", "mean_r", "mean_g", "mean_b"]:
        med[feat] = stats.groupby("class_name")[feat].median().reindex(CLASSES).round(4).to_dict()
    R["medians"] = med

    R["summaryRows"] = [
        ["Source", "Kaggle – Garbage Images Dataset (2000/class), version 5, MIT licence (Cofone, 2025)"],
        ["Total number of images", R["n_images_fmt"]],
        ["Number of classes", "6 – cardboard, glass, metal, paper, plastic, trash"],
        ["Images per class (files)", ", ".join(f"{c} {fmt(v)}" for c, v in counts.items())],
        ["Unique images per class", ", ".join(f"{c} {fmt(v)}" for c, v in g["unique"].items())],
        ["Image dimensions", "256 × 256 pixels (all images)"],
        ["File format / colour mode", "JPEG / 8-bit RGB (3 channels) – all images"],
        ["Total size / median file size", f"{R['total_mb']} MB / {R['median_kb']} KB"],
        ["Unreadable or corrupted files", str(summary["n_images"] - summary["n_readable"])],
        ["Metadata", "metadata.csv (filename, label): 100% agreement with the folders"],
        ["Class balance (largest/smallest): files / unique images", f"{R['imbalance_ratio']} / {R['dup']['ratio_unique']:.2f}"],
        ["Byte-identical duplicates", f"{R['exact_dup_files_fmt']} files ({R['exact_dup_pct']}) in {R['exact_dup_groups_fmt']} pairs"],
        ["Images with a near-duplicate (pHash ≤ 4)", f"{R['near_dup_images_fmt']} ({R['near_dup_pct']})"],
        ["Identical pairs with conflicting labels", str(R["cross_md5_groups"])],
    ]
    return stats


def provenance_part(R, m, stats):
    prov_sum, prov_img = m / "provenance_summary.csv", m / "provenance_per_image.csv"
    if not prov_sum.exists():
        R["provenance_text"] = ("The provenance check could not be run in this execution (it needs internet access "
                                "to Kaggle).")
        R["provenance_viz_text"] = R["provenance_text"]
        R["dup_origin_text"] = ""
        return
    ps = pd.read_csv(prov_sum, index_col=0).rename(columns=OLD_SOURCE_NAMES)
    total = ps.loc["total"]
    n = int(total.sum())
    sources = [c for c in ps.columns if c != "unidentified"]
    ident = int(total[sources].sum())
    order = sorted(sources, key=lambda c: -total[c])
    parts = [f"{SOURCE_SHORT.get(c, c)} ({fmt(total[c])} images, {pct(total[c] / n, 0)})" for c in order]
    R["provenance"] = {"total": {k: int(v) for k, v in total.items()}, "identified": ident, "n": n}
    per_class = ps.drop(index="total").reindex(CLASSES)
    share = per_class.div(per_class.sum(axis=1), axis=0)

    extra = ""
    trash_text = ""
    dup_text = ""
    if prov_img.exists():
        pi = pd.read_csv(prov_img).replace({"source": OLD_SOURCE_NAMES})
        pi = pi.merge(stats[["rel_path", "md5"]], on="rel_path", how="left")
        # where do the byte-identical pairs come from?
        dup_md5 = stats.groupby("md5").size()
        dup_md5 = set(dup_md5[dup_md5 > 1].index)
        dups = pi[pi["md5"].isin(dup_md5)]
        if len(dups):
            top = dups["source"].value_counts(normalize=True)
            src, sh = top.index[0], top.iloc[0]
            if src != "unidentified":
                dup_text = (f"The provenance analysis (Section 1.2.5) shows where the copies come from: "
                            f"{pct(sh, 0)} of the duplicated files match {SOURCE_SHORT.get(src, src)}, so many of "
                            "its photos appear twice – most likely because the creator merged two collections that "
                            "both contained them, or topped up the smaller classes with copies.")
        # which source classes became our 'trash' class?
        t = pi[(pi["class_name"] == "trash") & (pi["source"] != "unidentified")]
        if len(t):
            sc = t["source_class"].fillna("").str.lower().value_counts(normalize=True)
            sc = sc[sc >= 0.03]
            trash_text = (" The source folders of the matched *trash* images reveal what this class really contains: "
                          + join_and([f"{name} ({pct(v, 0)})" for name, v in sc.items()]) + ".")
            R["trash_sources"] = {k: round(float(v), 4) for k, v in sc.items()}
        R["prov_dist_median"] = float(pi.loc[pi["source"] != "unidentified", "distance"].median())
        extra = trash_text

    tn = "TrashNet (Thung & Yang, 2016)"
    recyc = ["cardboard", "glass", "metal", "paper", "plastic"]
    main_trash_src = share.loc["trash", sources].idxmax()
    trash_main_share = share.loc["trash", main_trash_src]
    if tn in share.columns:
        tn_share = share.loc[recyc, tn]
        tn_text = (f"TrashNet accounts for {pct(tn_share.min(), 0)}–{pct(tn_share.max(), 0)} of the images of each "
                   f"recyclable class (most for {tn_share.idxmax()}) but only {pct(share.loc['trash', tn], 0)} of the trash images")
    else:
        tn_text = "TrashNet could not be checked"
    almost = "almost entirely" if trash_main_share >= 0.8 else "mainly"
    R["provenance_text"] = (
        f"{fmt(ident)} of the {fmt(n)} images ({pct(ident / n, 0)}) could be matched to a public source: "
        f"{join_and(parts)}; the remaining {pct(total.get('unidentified', 0) / n, 0)} come from collections that "
        f"were not identified. {tn_text}, whereas the *trash* class was taken {almost} from "
        f"{SOURCE_SHORT.get(main_trash_src, main_trash_src)} ({pct(trash_main_share, 0)} of its images).")
    R["provenance_viz_text"] = (
        "Finally, ${figRef(\"provenance\")} shows the composition of each class by source data set, measured by "
        f"perceptual-hash matching against three public collections. {tn_text}; {pct(trash_main_share, 0)} of the "
        f"trash images match {SOURCE_SHORT.get(main_trash_src, main_trash_src)}. Because the classes were assembled "
        "from sources with different photographic styles, part of what distinguishes a class in this data set is its "
        "*source* (background, lighting, camera) rather than its material – a risk that the evaluation addresses."
        + extra)
    R["dup_origin_text"] = dup_text
    R["provenance_share"] = share.round(4).to_dict()


# ------------------------------------------------------------------------------------------
# Modelling and evaluation (Section 2)
# ------------------------------------------------------------------------------------------
PAIR_REASON = {
    frozenset(["paper", "cardboard"]): "both are flat, fibre-based and often printed, and thin cardboard and thick paper look almost the same",
    frozenset(["plastic", "glass"]): "transparent bottles and jars share shape, transparency and reflections",
    frozenset(["plastic", "metal"]): "shiny packaging, metallised films and caps reflect light like metal",
    frozenset(["glass", "metal"]): "both are hard and reflective, and many bottles have metal caps",
    frozenset(["plastic", "paper"]): "white or printed plastic film and bags resemble paper",
    frozenset(["paper", "trash"]): "dirty, crumpled or mixed paper items resemble the heterogeneous trash class",
    frozenset(["plastic", "trash"]): "crumpled plastic packaging resembles the mixed items of the trash class",
    frozenset(["cardboard", "trash"]): "food-stained or composite cartons resemble trash",
    frozenset(["metal", "trash"]): "small or dirty metal items are hard to tell from mixed rubbish",
    frozenset(["glass", "trash"]): "broken or dirty glass resembles rubbish",
    frozenset(["cardboard", "metal"]): "printed cans and boxes share labels and colours",
    frozenset(["cardboard", "plastic"]): "printed packaging of both materials looks alike",
    frozenset(["glass", "paper"]): "bottles with paper labels show both materials",
    frozenset(["cardboard", "glass"]): "brown glass and brown cardboard share colour",
    frozenset(["metal", "paper"]): "foil and metallic print look similar on flat items",
}


def evaluation_part(R, m, test, per_class, confused, selective, robust, hist, train_cfg):
    pc = per_class.set_index("class")
    best, worst = pc["f1"].idxmax(), pc["f1"].idxmin()
    rec_lo, prec_lo = pc["recall"].idxmin(), pc["precision"].idxmin()
    interp = {}
    interp["per_class"] = (
        f"All six classes reach an F1-score of at least {pc['f1'].min():.3f}, so no material is neglected. Recall – the "
        f"share of items of a class that are found – is lowest for **{rec_lo}** ({pc.loc[rec_lo, 'recall']:.3f}), "
        f"and precision is lowest for **{prec_lo}** ({pc.loc[prec_lo, 'precision']:.3f}), which means that items of "
        f"other materials are most often mistaken for {prec_lo}. "
        f"The {fmt(test['n'] * (1 - test['accuracy']))} errors are spread over many small off-diagonal cells rather than one systematic confusion.")

    top = confused.head(3)
    parts = []
    for r in top.itertuples():
        reason = PAIR_REASON.get(frozenset([r.true, r.predicted]), "the two materials look similar in some photos")
        parts.append(f"**{r.true} → {r.predicted}** ({r.count} images, {pct(r.share_of_true_class)} of {r.true}; {reason})")
    interp["confusion"] = (
        "The most frequent confusions (" + tabRef_placeholder("confused") + ") are " + join_and(parts) + ". "
        "These are exactly the visually similar pairs predicted by the data analysis (Section 1.2.4): the confusions "
        "are plausible mistakes between materials with similar appearance rather than random errors.")

    auc = pc["roc_auc"]
    interp["roc"] = (
        f"All one-vs-rest ROC curves run close to the top-left corner (AUC {auc.min():.3f}–{auc.max():.3f}; macro "
        f"{test['roc_auc_ovr_macro']:.3f}): for any material, a randomly chosen item of that material receives a "
        f"higher score than a randomly chosen other item in more than {100 * auc.min():.0f}% of cases. The lowest AUC "
        f"belongs to **{auc.idxmin()}**. The precision-recall curves stay high until recall approaches 1, so a "
        "class-specific threshold could trade a few missed items for very high precision if contamination of a bin "
        "had to be avoided.")

    T = test["temperature"]
    if T < 1:
        cal = ("Before scaling, the points lie *above* the diagonal: the model was **under-confident** – it was right "
               "more often than its probabilities claimed. This is a known side-effect of label smoothing, which "
               "deliberately keeps predicted probabilities away from 0 and 1. ")
    else:
        cal = ("Before scaling, the points lie *below* the diagonal: the model was **over-confident**, as is typical "
               "for modern deep networks (Guo et al., 2017). ")
    cal_detail = ""
    pred_csv = m / "test_predictions.csv"
    if pred_csv.exists():
        import numpy as np

        preds = pd.read_csv(pred_csv)
        prob = preds[[f"p_{c}" for c in CLASSES]].to_numpy(float)
        conf, correct = prob.max(1), prob.argmax(1) == preds["label"].to_numpy()
        hi = conf > 0.8
        mid = (conf > 0.4) & (conf <= 0.8)
        b89 = (conf > 0.8) & (conf <= 0.9)
        b9 = conf > 0.9
        cal_detail = (
            f"After scaling, the predictions with more than 80% confidence – {pct(hi.mean())} of the test images – are "
            f"almost perfectly calibrated (mean confidence {pct(conf[b89].mean())} vs accuracy {pct(correct[b89].mean())} "
            f"in the 80–90% bin, {pct(conf[b9].mean())} vs {pct(correct[b9].mean())} above 90%). The {int(mid.sum())} "
            f"predictions between 40% and 80% confidence are somewhat over-confident (mean confidence "
            f"{pct(conf[mid].mean(), 0)}, accuracy {pct(correct[mid].mean(), 0)}); these few bins also contain too few "
            "images for precise estimates. Moderately confident predictions should therefore be checked, which is why "
            "the application names the second most likely class and flags the least confident ones as UNCERTAIN.")
    interp["calibration"] = cal + (
        f"Temperature scaling (T = {T:.2f}, fitted on the validation set) moved the curve towards the diagonal and "
        f"reduced the expected calibration error from {test['ece_before_temperature']:.3f} to {test['ece']:.3f}, "
        f"and the log-loss from {test['log_loss_before_temperature']:.3f} to {test['log_loss']:.3f}. " + cal_detail)

    sel = selective.set_index("threshold")
    def at(t):
        idx = (sel.index.to_series() - t).abs().idxmin()
        return sel.loc[idx]
    s0, s6, s8, s9 = at(0.0), at(0.6), at(0.8), at(0.9)
    interp["selective"] = (
        f"Correct predictions are concentrated at very high confidence, whereas wrong predictions are spread over "
        f"lower values (left panel). Rejecting uncertain predictions therefore raises the accuracy of the accepted "
        f"ones (right panel): without a threshold all images are classified with {pct(s0.accuracy)} accuracy; with "
        f"the application's default threshold of 60%, {pct(s6.coverage)} of the images are accepted with "
        f"{pct(s6.accuracy)} accuracy; at 80% and 90% the accuracy rises to {pct(s8.accuracy)} and {pct(s9.accuracy)} "
        f"but {pct(1 - s8.coverage)} and {pct(1 - s9.coverage)} of the images are flagged. The 60% default was "
        "chosen as a compromise: it flags only a few percent of the images – the ones most likely to be wrong – "
        "without asking the user to check too many items, and the user can move the slider.")

    nice = {"rotated 90 deg": "90° rotation", "greyscale": "grey-scale", "blur": "blur", "Gaussian noise": "Gaussian noise",
            "JPEG quality 20": "strong JPEG compression (quality 20)", "darker (-40%)": "40% darker", "brighter (+40%)": "40% brighter"}
    rb = robust.set_index("corruption")["accuracy"].rename(index=lambda k: nice.get(k, k))
    orig = rb.loc["original"]
    drops = (orig - rb.drop(index="original")).sort_values()
    small = [k for k, v in drops.items() if v <= 0.03]
    mid = drops[(drops > 0.03) & (drops <= 0.05)]
    big = drops[drops > 0.05]
    txt = (f"On the unmodified sample the accuracy is {pct(orig)}. ")
    if small:
        txt += f"It changes by at most 3 percentage points for {join_and(small)}, "
        txt += "so typical variations of lighting, sensor noise and compression hardly matter. "
    if len(mid):
        txt += ("Moderate drops occur for " + join_and([f"{k} ({pct(rb.loc[k])})" for k in mid.index]) + ". ")
    if len(big):
        txt += ("The largest drop occurs for " if len(big) == 1 else "The largest drops occur for ")
        txt += join_and([f"{k} ({pct(rb.loc[k])})" for k in big.index[::-1]]) + ". "
    if "grey-scale" in big.index:
        txt += "Removing colour hurts because colour is a genuine cue (brown cardboard, green glass), as shown in Section 1.2.3. "
    if "90° rotation" in big.index or "90° rotation" in mid.index:
        txt += "A 90° rotation creates layouts (horizontal bottles, sideways text) that are rare in the training photos. "
    interp["robustness"] = txt + ("In the application, the confidence flag warns the user when a photo is unusual, "
                                   "and the user guide asks for upright, well-lit photos.")
    R["robust_weak"] = join_and([f"{k} ({pct(rb.loc[k])})" for k in drops.index[::-1][:2]])
    R["robust_orig_pct"] = pct(orig)

    interp["summary"] = [
        f"**Effective:** {pct(test['accuracy'])} accuracy and {test['macro_f1']:.3f} macro-F1 on {fmt(test['n'])} unseen, de-duplicated test images, with a narrow 95% confidence interval ({pct(test['accuracy_ci'][0])}–{pct(test['accuracy_ci'][1])}).",
        "**Better than the alternatives:** clearly above every classical model, the CNN trained from scratch and the frozen-feature linear probe (${tabRef(\"baselines\")}, ${figRef(\"comparison\")}).",
        f"**Balanced:** F1 between {pc['f1'].min():.3f} ({worst}) and {pc['f1'].max():.3f} ({best}); errors are plausible confusions between similar materials.",
        f"**Trustworthy confidence:** after temperature scaling the calibration error is {test['ece']:.3f}, so the UNCERTAIN flag is meaningful.",
        f"**Robust to everyday variation** (lighting, sensor noise, compression), less so to {R['robust_weak']}.",
        "**Limitations:** the test images come from the same sources as the training images; photos taken by users in their kitchens may be harder (see Conclusion).",
    ]
    R["interp"] = interp

    # ---- learning curves
    ph1, ph2 = hist[hist["phase"] == "head"], hist[hist["phase"] != "head"]
    best_epoch = int(train_cfg["best_epoch"])
    b = hist.loc[hist["epoch"] == best_epoch].iloc[0]
    stopped = len(ph2) < train_cfg["finetune_epochs"]
    last = hist.iloc[-1]
    vmin_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    rises = bool((hist.loc[hist["epoch"] > vmin_epoch, "val_loss"] > hist["val_loss"].min() + 0.02).any())
    gap = float(last["train_acc"] - last["val_acc"])
    R["learning_text"] = (
        f"During phase 1 (frozen backbone) the validation accuracy reaches {pct(ph1['val_acc'].max())} within "
        f"{len(ph1)} epochs. As soon as all layers are unfrozen (phase 2) it jumps to {pct(ph2['val_acc'].iloc[0])} in the "
        f"first fine-tuning epoch and then improves steadily while the learning rate decays. The best validation "
        f"macro-F1 ({b['val_macro_f1']:.3f}, accuracy {pct(b['val_acc'])}) was reached in epoch {best_epoch} of {len(hist)}"
        + (", after which early stopping ended training. " if stopped else "; by then the validation curves had flattened, so more epochs would have brought little gain. ")
        + f"In the last epochs the training accuracy ({pct(last['train_acc'])}) is {100 * gap:.1f} percentage points above the "
        f"validation accuracy ({pct(last['val_acc'])}): the network fits the training photos better than new ones, as is normal "
        "for a model of four million parameters. "
        + ("However, the validation loss never rises again after its minimum – it only flattens – so this gap does not "
           "harm generalisation, and the regularisation (augmentation, dropout, drop-path, label smoothing, weight decay) "
           "together with model selection on the validation set kept over-fitting under control. "
           if not rises else
           "The validation loss rises again after epoch " + str(vmin_epoch) + ", a sign of over-fitting that early stopping "
           "counteracts by keeping the best epoch. ")
        + f"The training loss stays at about {last['train_loss']:.2f} instead of approaching zero because label smoothing "
        "keeps the targets at 0.9 instead of 1.")


PHOTO_FIGS = {"fig02_sample_images", "fig07_mean_images", "fig11_duplicate_examples", "fig14_augmentation_examples",
              "fig21_misclassified_examples", "fig22_gradcam"}


def cache_figures(src: Path, dst: Path, max_w: int = 1700) -> Path:
    """Copy every PNG of `src` to `dst`, at most `max_w` pixels wide. Photo-like figures are stored as
    high-quality JPEG (much smaller than PNG); charts stay PNG so that lines and text remain sharp."""
    from PIL import Image

    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.png")):
        photo = f.stem in PHOTO_FIGS
        out = dst / (f.stem + (".jpg" if photo else ".png"))
        if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            continue
        img = Image.open(f).convert("RGB")
        if img.width > max_w:
            img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
        if photo:
            img.save(out, "JPEG", quality=88, optimize=True)
        else:
            img.save(out, "PNG", optimize=True)
    return dst


def tabRef_placeholder(key):
    # replaced in the JavaScript builder by the real table number
    return "${tabRef(\"" + key + "\")}"


def iteration_part(R, prev_dir, test, robust, hist, train_cfg):
    if not prev_dir:
        return
    pm = Path(prev_dir) / "metrics"
    ptest = json.loads((pm / "test_metrics.json").read_text())
    prob = pd.read_csv(pm / "robustness.csv").set_index("corruption")["accuracy"]
    phist = pd.read_csv(pm / "history_efficientnet_b0.csv")
    pcfg = json.loads((pm / "training_config.json").read_text())
    rb = robust.set_index("corruption")["accuracy"]

    def row(name, a, b, kind="pct"):
        if kind == "pct":
            return [name, pct(a), pct(b), f"{100 * (b - a):+.1f} points"]
        if kind == "f3":
            return [name, f"{a:.3f}", f"{b:.3f}", "unchanged" if abs(b - a) < 0.0005 else f"{b - a:+.3f}"]
        return [name, str(a), str(b), ""]

    p1_prev = phist.loc[phist["phase"] == "head", "val_acc"].max()
    p1_now = hist.loc[hist["phase"] == "head", "val_acc"].max()
    rows = [
        row("Best validation accuracy in phase 1 (head only)", p1_prev, p1_now),
        row("Test accuracy", ptest["accuracy"], test["accuracy"]),
        row("Test macro-F1", ptest["macro_f1"], test["macro_f1"], "f3"),
        row("Accuracy with Gaussian noise", prob.get("Gaussian noise", float("nan")), rb.get("Gaussian noise", float("nan"))),
        row("Accuracy with blur", prob.get("blur", float("nan")), rb.get("blur", float("nan"))),
        row("Accuracy in grey-scale", prob.get("greyscale", float("nan")), rb.get("greyscale", float("nan"))),
        row("Expected calibration error (after scaling)", ptest["ece"], test["ece"], "f3"),
        ["Best epoch / epochs trained", f"{pcfg['best_epoch']} / {len(phist)}", f"{train_cfg['best_epoch']} / {len(hist)}", ""],
    ]
    first_loss = phist["train_loss"].iloc[0]
    R["iteration"] = {
        "rows": rows,
        "note": "Both runs used the same de-duplicated split, test set and evaluation code. Robustness values are measured on the same stratified sample of test images.",
        "text": (
            "Model development was iterative. The first complete run (run 1) already reached "
            f"{pct(ptest['accuracy'])} test accuracy, but its evaluation revealed three weaknesses. (1) Phase 1 "
            f"reached only {pct(p1_prev)} validation accuracy, although a simple logistic regression on the same "
            f"frozen features reached about 90% (${{tabRef(\"baselines\")}}); the training loss of the first epoch "
            f"({first_loss:.2f}) was far above ln 6 = 1.79, showing that the randomly initialised layer produced "
            "large, wrong scores that wasted the head-training phase. (2) Accuracy collapsed to "
            f"{pct(prob.get('Gaussian noise', float('nan')))} on images with sensor noise. (3) The best epoch was "
            "the last one, so training had not fully converged. Run 2 therefore initialised the new layer to zero, "
            "added blur and noise to the augmentation (and removed the small hue change), and allowed three more "
            "fine-tuning epochs. The table compares both runs; run 2 is the model used in the application and "
            "reported in Section 2.4."),
    }


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("out_json")
    ap.add_argument("--previous", default=None)
    ap.add_argument("--manual", default=str(HERE / "manual_text.json"))
    ap.add_argument("--app-fig-dir", default=str(PROJECT / "reports" / "figures" / "app"))
    ap.add_argument("--static-fig-dir", default=str(PROJECT / "reports" / "figures"))
    ap.add_argument("--zip-code", default="smart-waste-sorter-code.zip")
    ap.add_argument("--zip-data", default="smart-waste-sorter-data-and-model.zip")
    ap.add_argument("--word-count", default="")
    args = ap.parse_args()

    rd = Path(args.results_dir).resolve()
    m = rd / "metrics"
    summary = json.loads((m / "dataset_summary.json").read_text())
    test = json.loads((m / "test_metrics.json").read_text())
    info = json.loads((rd / "model" / "model_info.json").read_text())
    train_cfg = json.loads((m / "training_config.json").read_text())
    run_info = json.loads((m / "run_info.json").read_text()) if (m / "run_info.json").exists() else {}
    comp = pd.read_csv(m / "model_comparison.csv")
    per_class = pd.read_csv(m / "per_class_metrics.csv")
    hist = pd.read_csv(m / "history_efficientnet_b0.csv")
    split_counts = pd.read_csv(m / "split_counts.csv", index_col=0)
    robust = pd.read_csv(m / "robustness.csv")
    confused = pd.read_csv(m / "most_confused_pairs.csv")
    selective = pd.read_csv(m / "selective_prediction.csv")

    R = {"results_dir": str(rd), "app_fig_dir": str(Path(args.app_fig_dir).resolve()),
         "static_fig_dir": str(Path(args.static_fig_dir).resolve()),
         "zip_code": args.zip_code, "zip_data": args.zip_data,
         "wordCountLabel": args.word_count or "—"}
    stats = dataset_part(R, m, summary)
    provenance_part(R, m, stats)

    n_model = int(split_counts.loc["total"].sum())
    R["n_model"] = n_model
    R["n_model_fmt"] = fmt(n_model)
    R["n_test_fmt"] = fmt(test["n"])
    R["splitRows"] = []
    for c in CLASSES + ["total"]:
        r = split_counts.loc[c]
        R["splitRows"].append([c if c != "total" else "Total", fmt(r["train"]), fmt(r["val"]), fmt(r["test"]),
                               fmt(r["train"] + r["val"] + r["test"])])

    R.update(
        acc=test["accuracy"], acc_pct=pct(test["accuracy"]),
        acc_ci=f"{pct(test['accuracy_ci'][0])}–{pct(test['accuracy_ci'][1])}",
        f1_3=f"{test['macro_f1']:.3f}", f1_ci=f"{test['macro_f1_ci'][0]:.3f}–{test['macro_f1_ci'][1]:.3f}",
        bal_acc_pct=pct(test["balanced_accuracy"]), top2_pct=pct(test["top2_accuracy"]),
        auc_3=f"{test['roc_auc_ovr_macro']:.3f}", kappa_3=f"{test['cohen_kappa']:.3f}", mcc_3=f"{test['mcc']:.3f}",
        logloss_3=f"{test['log_loss']:.3f}", logloss_before_3=f"{test['log_loss_before_temperature']:.3f}",
        ece_before_3=f"{test['ece_before_temperature']:.3f}", ece_3=f"{test['ece']:.3f}",
        temperature_2=f"{test['temperature']:.2f}",
        macro_p_3=f"{test['macro_precision']:.3f}", macro_r_3=f"{test['macro_recall']:.3f}",
        coverage_pct=pct(test["coverage_above_threshold"]), acc_above_pct=pct(test["accuracy_above_threshold"]),
        n_wrong=int(round(test["n"] * (1 - test["accuracy"]))),
        val_acc_pct=pct(test["val"]["accuracy"]), val_f1_3=f"{test['val']['macro_f1']:.3f}",
    )
    R["test_metrics"] = test
    R["per_class"] = per_class.round(4).to_dict("records")
    R["confused"] = confused.round(4).to_dict("records")
    R["selective"] = selective.round(4).to_dict("records")
    R["robust"] = robust.round(4).to_dict("records")

    # ---- baselines table
    order = ["k-NN (k=5) + HOG/colour/LBP", "Logistic regression + HOG/colour/LBP", "Random forest + HOG/colour/LBP",
             "SVM (RBF) + HOG/colour/LBP", "Small CNN trained from scratch", "Frozen EfficientNet-B0 + logistic regression",
             "EfficientNet-B0 fine-tuned (final)"]
    comp = comp.assign(order=comp["model"].map({k: i for i, k in enumerate(order)})).sort_values("order")

    def secs(s):
        return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"
    R["baselineRows"] = [[r.model, r.family, pct(r.val_accuracy), pct(r.test_accuracy), f"{r.test_macro_f1:.3f}",
                          secs(r.fit_seconds)] for r in comp.itertuples()]
    classical = comp[comp["family"] == "classical ML"]
    best_c = classical.loc[classical["test_accuracy"].idxmax()]
    R["best_classical_name"] = best_c["model"].split(" +")[0]
    R["best_classical_acc"] = pct(best_c["test_accuracy"])
    R["classical_range"] = f"{pct(classical['test_accuracy'].min())}–{pct(classical['test_accuracy'].max())}"
    cnn = comp[comp["model"].str.contains("scratch")]
    R["cnn_acc"] = pct(cnn["test_accuracy"].iloc[0]) if len(cnn) else "n/a"
    probe = comp[comp["model"].str.contains("Frozen")]
    R["probe_acc"] = pct(probe["test_accuracy"].iloc[0]) if len(probe) else "n/a"

    # ---- training
    R["cfg"] = {k: train_cfg[k] for k in ["head_epochs", "head_lr", "finetune_epochs", "finetune_lr", "weight_decay",
                                          "batch_size", "patience", "label_smoothing"]}
    R["best_epoch"] = int(train_cfg["best_epoch"])
    R["n_params_fmt"] = fmt(train_cfg["n_parameters"])
    R["head_params_fmt"] = fmt(1280 * 6 + 6)
    R["train_time"] = f"{train_cfg['train_seconds'] / 60:.0f} minutes ({len(hist)} epochs)"
    R["history"] = hist.round(4).to_dict("records")
    evaluation_part(R, m, test, per_class, confused, selective, robust, hist, train_cfg)
    iteration_part(R, args.previous, test, robust, hist, train_cfg)

    # ---- export / environment
    R["onnx_mb"] = f"{info['onnx_file_mb']:.1f}"
    R["latency_ms"] = f"{info['cpu_latency_ms']['median_ms']:.0f}"
    R["parity_n"] = str(info["parity"]["n_images_checked"])
    R["parity_agree"] = pct(info["parity"]["prediction_agreement_fp16_weights"], 1)
    R["parity_diff"] = f"{info['parity']['max_abs_diff_fp16_weights']:.3f}"
    env = info.get("environment", {})
    R["env"] = {"python": env.get("python", "3"), "torch": env.get("torch", "").split("+")[0],
                "timm": env.get("timm", ""), "gpu": env.get("gpu", "T4").replace("Tesla ", ""),
                "sklearn": env.get("sklearn", ""), "numpy": env.get("numpy", ""), "pandas": env.get("pandas", ""),
                "onnxruntime": env.get("onnxruntime", ""), "onnx": env.get("onnx", ""), "cv2": env.get("cv2", ""),
                "skimage": env.get("skimage", ""), "matplotlib": env.get("matplotlib", ""), "PIL": env.get("PIL", ""),
                "torchvision": env.get("torchvision", "").split("+")[0], "imagehash": env.get("imagehash", "")}
    R["timings"] = run_info.get("timings_s", {})
    app_env = json.loads((HERE / "app_env.json").read_text()) if (HERE / "app_env.json").exists() else {}
    E = R["env"]
    R["libraryRows"] = [
        ["Streamlit", app_env.get("streamlit", "1.64"), "Web user interface of the application (Snowflake Inc., 2026)"],
        ["ONNX Runtime", app_env.get("onnxruntime", E["onnxruntime"]), "Fast CPU inference of the exported model (ONNX Runtime developers, 2026)"],
        ["NumPy", app_env.get("numpy", E["numpy"]), "Array maths: pre-processing, soft-max"],
        ["Pillow (PIL)", app_env.get("pillow", E["PIL"]), "Reading, validating, converting and resizing images"],
        ["pandas", app_env.get("pandas", E["pandas"]), "Result tables and CSV export"],
        ["PyTorch / torchvision", f"{E['torch']} / {E['torchvision']}", "Training the neural networks, data loading and augmentation (Paszke et al., 2019)"],
        ["timm", E["timm"], "Pre-trained EfficientNet-B0 (Wightman, 2019)"],
        ["scikit-learn", E["sklearn"], "Baseline models, PCA, splits, metrics (Pedregosa et al., 2011)"],
        ["scikit-image / OpenCV", f"{E['skimage']} / {E['cv2']}", "HOG and LBP features, edges, image statistics"],
        ["ImageHash", E["imagehash"], "Perceptual hashes for duplicate and provenance analysis"],
        ["Matplotlib", E["matplotlib"], "All figures of the report"],
        ["ONNX", E["onnx"], "Model export and fp16 weight compression"],
        ["kagglehub", "0.3+", "Downloading the data sets in Colab"],
    ]

    # ---- downscaled copies of all figures (keeps the .docx small without visible loss)
    cache = Path(args.out_json).resolve().parent / "fig_cache"
    R["fig_dir"] = str(cache_figures(rd / "figures", cache / "results"))
    R["static_fig_dir"] = str(cache_figures(Path(R["static_fig_dir"]), cache / "static"))
    R["app_fig_dir"] = str(cache_figures(Path(R["app_fig_dir"]), cache / "app", max_w=1500))

    manual = Path(args.manual)
    if manual.exists():
        R["interp"].update(json.loads(manual.read_text()))
    for key in ["errors", "gradcam"]:
        R["interp"].setdefault(key, "")
    # resolve table/figure placeholders later in JS: keep as template text
    Path(args.out_json).write_text(json.dumps(R, indent=1, default=str))
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
