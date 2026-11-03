import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    BLUE,
    INK,
    MUTED,
    ORANGE,
    latent_inputs,
    load_net,
    neurips,
    open_run,
    output_dir,
    save,
)

BINS = 100
ENCODER, DECODER = BLUE, ORANGE
COLUMNS = 4


def short(label):
    """The label of a stage for a panel title."""
    label = (
        label.replace("decoder: ", "")
        .replace("pre-activation", "pre-act.")
        .replace("activation", "act.")
        .replace("inverse act.", "inv. act.")
    )
    return (
        label.replace("Arctan2 inverse (unit pairs)", "Arctan2 inverse")
        .replace("layer 1 output (reconstruction)", "reconstruction")
        .replace("(output z)", "(z)")
        .replace("radius-scaled pairs", "radius-scaled")
    )


def panels(labels):
    """The stages of a block as panels: the pre-activation and the activation of
    an encoder layer share one panel, every other stage has its own."""
    out, j = [], 0
    while j < len(labels):
        if (
            labels[j].endswith("pre-activation")
            and j + 1 < len(labels)
            and labels[j + 1] == labels[j].replace("pre-activation", "activation")
        ):
            out.append(labels[j : j + 2])
            j += 2
        else:
            out.append(labels[j : j + 1])
            j += 1
    return out


@neurips
def plot_distributions(out, stats, data):
    """One panel per stage (encoder block, decoder block), density histograms of
    the pooled elements; an encoder layer's pre-activation (grey) and activation
    (colour) share a panel."""
    groups = [
        (
            "Encoder",
            ENCODER,
            panels([l for l in stats if not l.startswith("decoder:")]),
        ),
        ("Decoder", DECODER, panels([l for l in stats if l.startswith("decoder:")])),
    ]
    rows = [-(-len(g[2]) // COLUMNS) for g in groups]
    fig, axes = plt.subplots(
        sum(rows), COLUMNS, figsize=(5.5, 1.3 * sum(rows) + 0.1), squeeze=False
    )
    r0 = 0
    for (name, color, plist), nrows in zip(groups, rows):
        for ax in axes[r0 : r0 + nrows].ravel():
            ax.axis("off")
        for ax, labels in zip(axes[r0 : r0 + nrows].ravel(), plist):
            ax.axis("on")
            top = 0
            pair = len(labels) == 2
            for k, label in enumerate(labels):
                c = color if k == len(labels) - 1 else MUTED
                counts, edges = data[f"{label}_counts"], data[f"{label}_edges"]
                dens = counts / (counts.sum() * np.diff(edges))
                top = max(top, dens.max())
                ax.stairs(dens, edges, fill=True, color=c, alpha=0.3, lw=0)
                ax.stairs(dens, edges, color=c, lw=0.8)
                m = stats[label]
                tag = ("pre-act. " if k == 0 else "act. ") if pair else ""
                ax.text(
                    0,
                    1.03 + 0.17 * (len(labels) - 1 - k),
                    f"{tag}$\\mu$ = {m['mean']:.2g}, $\\sigma$ = {m['std']:.2g}"
                    + ("" if pair else f", $d$ = {m['width']}"),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=5.5,
                    color=c,
                )
            lo, hi = (
                min(data[f"{l}_edges"][0] for l in labels),
                max(data[f"{l}_edges"][-1] for l in labels),
            )
            if lo < 0 < hi:
                ax.axvline(0, color=MUTED, lw=0.4, ls=(0, (2, 2)))
            title = short(labels[-1])
            if "(z - b" in labels[-1]:
                i = labels[-1].split("b")[1][0]
                title = f"$(z - b_{{{i}}})\\,W_{{{i}}}^\\top$"
            if pair:
                title = (
                    title.replace(" act.", "")
                    + f" ($d$ = {stats[labels[-1]]['width']})"
                )
            ax.text(
                0,
                1.2 + 0.17 * (len(labels) - 1),
                title,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
            )
            ax.set_yticks([])
            ax.set_ylim(0, 1.08 * top)
            ax.set_xlim(lo, hi)
            ax.spines[["top", "right", "left"]].set_visible(False)
        for ax in axes[r0 : r0 + nrows, 0]:
            ax.set_ylabel("density")
        r0 += nrows
    fig.tight_layout(pad=0.3, h_pad=2.9, w_pad=0.6, rect=[0.035, 0, 1, 1])
    r0 = 0
    for (name, color, _), nrows in zip(groups, rows):
        top, bottom = (
            axes[r0, 0].get_position().y1,
            axes[r0 + nrows - 1, 0].get_position().y0,
        )
        fig.text(
            0.012,
            (top + bottom) / 2,
            name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
        )
        r0 += nrows
    save(fig, out, "activation_distributions")


def activation_distributions(run, out=None):
    run, cfg = open_run(run)
    out = output_dir("activation_distributions", run, out)
    net = load_net(run, cfg).eval()
    x = latent_inputs(cfg)[2]
    with torch.no_grad():
        stages = net.encoder_activations(x)
        assert torch.allclose(stages[-1][1], net.encode(x), atol=1e-5)
        dec = net.decoder_activations(stages[-1][1])
        assert torch.allclose(dec[-1][1], net.decode(stages[-1][1]), atol=1e-5)
        stages += dec

    stats, data = {}, {}
    for label, t in stages:
        v = t.detach().double().flatten().numpy()
        lo, hi = np.percentile(v, [0.1, 99.9])
        counts, edges = np.histogram(v, bins=BINS, range=(lo, hi) if hi > lo else None)
        stats[label] = dict(width=t.shape[-1], mean=float(v.mean()), std=float(v.std()))
        data[f"{label}_counts"], data[f"{label}_edges"] = counts, edges
    plot_distributions(out, stats, data)


if __name__ == "__main__":
    activation_distributions(*sys.argv[1:3])
