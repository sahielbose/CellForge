"""Generate the README figures as light and dark SVGs.

    python scripts/make_figures.py          # writes into figs/
"""

import math
import sys
from pathlib import Path

SANS = "-apple-system, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {
        "ink": "#1f2328", "muted": "#59636e", "faint": "#818b98", "line": "#d1d9e0", "panel": "#f6f8fa",
        "accent": "#0b6e6e", "accent_soft": "#d3ecea", "phase": ["#3554b8", "#6a3fb5", "#0b6e6e"],
        "ramp": ["#0b6e6e", "#3f9a96", "#8cc5c0"], "train": "#d8dee4", "control": "#8c959f",
        "behind": "#b4561c", "low": "#e8f3f2", "on_dark": "#ffffff",
    },
    "dark": {
        "ink": "#e6edf3", "muted": "#9198a1", "faint": "#6e7681", "line": "#3d444d", "panel": "#151b23",
        "accent": "#4fc4b8", "accent_soft": "#173a38", "phase": ["#7d9cf0", "#ab8bf0", "#4fc4b8"],
        "ramp": ["#6fd9cc", "#3c9e95", "#256a64"], "train": "#2f363d", "control": "#6e7681",
        "behind": "#f0a46a", "low": "#1b2a2c", "on_dark": "#0d1117",
    },
}

# Read from the paper's Figure 3g and 4g (overall rank, aggregated over eight metrics) and Figure 3f and 4f
# (AUROC of DE-gene recovery). None marks a method that was not run on that readout.
READOUTS = [("Adamson", "CRISPRi"), ("Norman", "CRISPRa"), ("Srivatsan", "drugs"), ("Schiebinger", "cytokines"),
            ("Papalexi", "RNA"), ("Papalexi", "protein"), ("Liscovitch", "ATAC")]
_ = None
OVERALL_RANK = {
    "CellForge": [3, 3, 4, 1, 1, 1, 1],
    "Linear": [2, 1, 1, 3, 2, 4, 2],
    "RF": [1, 2, 2, 5, 3, 2, 3],
    "Unperturbed": [4, 6, 3, 6, 5, 5, 6],
    "CellFlow": [5, 4, 5, 7, _, _, _],
    "STATE": [7, 5, 6, _, _, _, _],
    "GEARS": [6, 7, _, _, _, _, _],
    "scGPT": [8, 8, _, _, _, _, _],
    "Biolord": [9, 9, 7, _, _, _, _],
    "CPA": [10, 10, 8, _, _, _, _],
    "ChemCPA": [_, _, 8, _, _, _, _],
    "GNN": [_, _, _, 2, 4, 3, 4],
    "Transformer": [_, _, _, 4, 6, 6, 5],
}
AUROC = {
    "CellForge": [0.954, 0.965, 0.973, 0.986, 0.906, 0.619, 0.796],
    "Linear": [0.941, 0.963, 0.981, 0.980, 0.902, 0.606, 0.792],
    "RF": [0.946, 0.962, 0.983, 0.974, 0.904, 0.595, 0.793],
    "Unperturbed": [0.919, 0.926, 0.977, 0.953, 0.883, 0.490, 0.627],
    "CellFlow": [0.927, 0.952, 0.980, 0.960, _, _, _],
    "STATE": [0.935, 0.966, 0.955, _, _, _, _],
    "GEARS": [0.940, 0.952, _, _, _, _, _],
    "scGPT": [0.934, 0.935, _, _, _, _, _],
    "Biolord": [0.881, 0.903, 0.754, _, _, _, _],
    "CPA": [0.937, 0.934, 0.779, _, _, _, _],
    "ChemCPA": [_, _, 0.775, _, _, _, _],
    "GNN": [_, _, _, 0.980, 0.871, 0.606, 0.785],
    "Transformer": [_, _, _, 0.958, 0.771, 0.570, 0.734],
}


class Canvas:
    def __init__(self, width, height, theme, label):
        self.w, self.h, self.t, self.label = width, height, THEMES[theme], label
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, color="ink", weight=400, anchor="start", mono=False, italic=False):
        family = MONO if mono else SANS
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{self.t.get(color, color)}" text-anchor="{anchor}"{style}>{s}</text>')

    def rect(self, x, y, w, h, fill="none", stroke="line", r=6, width=1, dash=None):
        fill = self.t.get(fill, fill)
        stroke = self.t.get(stroke, stroke)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"{d}/>')

    def line(self, x1, y1, x2, y2, color="muted", width=1.25, arrow=False, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#arrow-{color})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.t[color]}" stroke-width="{width}"{d}{m}/>')

    def path(self, d, color="muted", width=1.25, arrow=False, dash=None, fill="none"):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#arrow-{color})"' if arrow else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{self.t[color]}" stroke-width="{width}"{dd}{m}/>')

    def box(self, x, y, w, h, title, sub=None, title_mono=False, fill="panel", stroke="line", sub2=None):
        self.rect(x, y, w, h, fill=fill, stroke=stroke)
        lines = 1 + bool(sub) + bool(sub2)
        top = y + h / 2 - (lines - 1) * 8 + 4
        self.text(x + w / 2, top, title, 12.5, weight=600, anchor="middle", mono=title_mono)
        if sub:
            self.text(x + w / 2, top + 16, sub, 11, color="muted", anchor="middle")
        if sub2:
            self.text(x + w / 2, top + 31, sub2, 11, color="muted", anchor="middle")

    def doc(self, x, y, w, h, name, sub):
        f = 10
        t = self.t
        self.add(f'<path d="M{x},{y} H{x + w - f} L{x + w},{y + f} V{y + h} H{x} Z" fill="{t["panel"]}" '
                 f'stroke="{t["ink"]}" stroke-width="1"/>')
        self.add(f'<path d="M{x + w - f},{y} V{y + f} H{x + w}" fill="none" stroke="{t["ink"]}" stroke-width="1"/>')
        self.text(x + w / 2, y + h / 2 - 2, name, 12, weight=600, anchor="middle", mono=True)
        self.text(x + w / 2, y + h / 2 + 14, sub, 11, color="muted", anchor="middle")

    def svg(self):
        markers = "".join(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{self.t[name]}"/></marker>'
            for name in ("muted", "accent", "ink", "faint")
        )
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{self.label}"><defs>{markers}</defs>'
                + "".join(self.parts) + "</svg>\n")


def phase_header(c, x, w, number, name, module, color):
    c.text(x, 34, f"{number}", 20, weight=700, color=color)
    c.text(x + 22, 34, name, 17, weight=650)
    c.text(x + w, 34, module, 11.5, color="muted", anchor="end", mono=True)
    c.add(f'<rect x="{x}" y="46" width="{w}" height="2.5" fill="{color}" rx="1"/>')


def pipeline(theme):
    c = Canvas(1200, 500, theme, "CellForge pipeline: formulation, ideation and execution, from dataset to scored model")
    t = c.t
    cols = [24, 416, 808]
    W = 368
    for i, (name, module) in enumerate([("Formulation", "Task_Analysis/"), ("Ideation", "Method_Design/"),
                                        ("Execution", "Code_Generation/ + autorun/")]):
        phase_header(c, cols[i], W, i + 1, name, module, t["phase"][i])

    # Formulation
    x = cols[0]
    for j, (title, sub) in enumerate([(".h5ad file", "facts read from disk"), ("task text", "the question"),
                                      ("literature", "corpus + PubMed")]):
        bx = x + j * 129
        c.box(bx, 68, 110, 50, title, sub, title_mono=(j == 0))
        c.line(bx + 55, 118, bx + 55, 146, arrow=True)
    for j, (a, b) in enumerate([("Dataset", "analyst"), ("Problem", "investigator"), ("Baseline", "assessor")]):
        bx = x + j * 129
        c.box(bx, 150, 110, 56, a, b)
        c.line(bx + 55, 206, x + 184, 240, arrow=True)
    c.box(x, 244, W, 58, "Refinement agent", "reviews all three, revises the flagged sections,",
          sub2="then writes recommendations (2 rounds max)")
    c.path(f"M{x + W},{262} C{x + W + 16},{262} {x + W + 16},{286} {x + W},{286}", arrow=True)
    c.line(x + W / 2, 302, x + W / 2, 336, arrow=True)
    c.doc(x + 74, 340, 220, 52, "task_analysis.json", "dataset, problem, baselines, plan")

    # Ideation
    x = cols[1]
    names = ["Data", "Biology", "Modeling", "Training"]
    centers = [x + 46 + k * 92 for k in range(4)]
    py = 152
    for a in range(4):
        for b in range(a + 1, 4):
            x1, x2 = centers[a], centers[b]
            h = 10 + 13 * (b - a)
            c.path(f"M{x1},{py - 14} C{x1},{py - 14 - h} {x2},{py - 14 - h} {x2},{py - 14}",
                   color="faint", width=1, dash="3 3")
    for k, name in enumerate(names):
        c.rect(centers[k] - 40, py - 14, 80, 28, fill="panel", stroke="ink", r=14)
        c.text(centers[k], py + 4.5, name, 12, weight=600, anchor="middle")
    c.text(x + W / 2, 82, "peer review between every pair", 10.5, color="faint", anchor="middle", italic=True)
    critic_y = 214
    for k in range(4):
        c.line(centers[k], py + 14, x + W / 2 + (k - 1.5) * 30, critic_y, color="accent", width=1.25)
    c.rect(x + W / 2 - 98, critic_y, 196, 30, fill="accent_soft", stroke="accent")
    c.text(x + W / 2, critic_y + 19.5, "Critic scores every proposal", 11.5, weight=600, anchor="middle")
    c.rect(x, 262, W, 58, fill="panel", stroke="line")
    c.add(f'<text x="{x + W / 2}" y="287" font-family="{MONO}" font-size="11.5" fill="{t["ink"]}" text-anchor="middle">'
          f'c<tspan font-size="9" dy="3">t</tspan><tspan dy="-3"> = 0.3·c</tspan><tspan font-size="9" dy="3">t−1</tspan>'
          f'<tspan dy="-3"> + 0.4·critic + 0.3·mean(peer)</tspan></text>')
    c.text(x + W / 2, 306, "stop when every c ≥ 0.8 and the spread is &lt; 0.03", 11, color="muted", anchor="middle")
    c.path(f"M{x + W},{291} C{x + W + 14},{291} {x + W + 14},{py} {x + W},{py}", arrow=True)
    c.text(x + W - 6, 246, "next round", 10.5, color="faint", anchor="end", italic=True)
    c.line(x + W / 2, 320, x + W / 2, 336, arrow=True)
    c.doc(x + 74, 340, 220, 52, "research_plan.json", "architecture, training, metrics")

    # Execution
    x = cols[2]
    c.box(x, 68, W, 46, "Codex writes the training script", "one task block of the plan at a time")
    c.line(x + W / 2 - 40, 114, x + W / 2 - 40, 134, arrow=True)
    c.line(x + W / 2 + 40, 138, x + W / 2 + 40, 118, arrow=True)
    c.text(x + W / 2 + 48, 130, "repair with the failure report", 10.5, color="faint", italic=True)
    c.box(x, 138, W, 50, "Verifier", "compiles, checks --help, and can run the script on control cells")
    c.line(x + W / 2, 188, x + W / 2, 208, arrow=True)
    # split strip
    c.rect(x, 212, W, 60, fill="panel", stroke="line")
    c.text(x + 12, 230, "Split", 12.5, weight=600)
    c.text(x + W - 12, 230, "five folds over perturbations", 11, color="muted", anchor="end")
    sx = x + 12
    for k in range(15):
        kind = "t" if k % 5 == 0 else ("v" if k == 1 else "r")
        fill = {"t": "accent", "v": "panel", "r": "train"}[kind]
        stroke = {"t": "accent", "v": "accent", "r": "train"}[kind]
        c.rect(sx + k * 19, 242, 15, 15, fill=fill, stroke=stroke, r=2, width=1.25)
    c.rect(sx + 15 * 19 + 10, 242, 46, 15, fill="control", stroke="control", r=2)
    c.text(sx + 15 * 19 + 33, 253.5, "ctrl", 9.5, color="panel", anchor="middle", weight=600)
    c.line(x + W / 2, 272, x + W / 2, 292, arrow=True)
    c.box(x, 296, W, 46, "Train, then predict held-out perturbations", "writes predictions.h5ad, never sees test cells")
    c.line(x + W / 2, 342, x + W / 2, 362, arrow=True)
    c.rect(x, 366, W, 62, fill="accent_soft", stroke="accent")
    c.text(x + W / 2, 388, "Harness scorer", 12.5, weight=600, anchor="middle")
    c.text(x + W / 2, 404, "MSE, PCC, R², DE metrics, Recall, AUROC", 11, color="muted", anchor="middle")
    c.text(x + W / 2, 419, "against no-change, mean-shift and linear baselines", 11, color="muted", anchor="middle")
    c.line(x + W / 2, 428, x + W / 2, 444, arrow=True)
    # the opt-in refinement loop: selection scores go back to the coding agent
    c.path(f"M{x + W},{386} C{x + W + 16},{386} {x + W + 16},{91} {x + W},{91}", color="accent", arrow=True,
           dash="4 3")
    ry = 240
    c.add(f'<text x="{x + W + 5}" y="{ry}" font-family="{SANS}" font-size="10.5" font-style="italic" '
          f'fill="{t["accent"]}" text-anchor="middle" transform="rotate(-90 {x + W + 5} {ry})" '
          f'paint-order="stroke" stroke="{t["panel"] if theme == "dark" else "#ffffff"}" stroke-width="4">'
          f'refinement loop, opt-in</text>')
    c.doc(x + 94, 448, 180, 44, "test_scores.json", "per held-out fold")

    # hand-offs between phases, and the legend
    for x0 in (cols[0] + W, cols[1] + W):
        c.line(x0 + 3, 47, x0 + 21, 47, color="ink", width=1.5, arrow=True)
    ly = 438
    c.rect(24, ly - 11, 22, 14, fill="accent_soft", stroke="accent", r=3)
    c.text(54, ly, "judges: the critic scores plans, the harness scores trained models", 11.5, color="muted")
    c.line(24, ly + 24, 46, ly + 24, color="faint", dash="3 3")
    c.text(54, ly + 28, "peer review between experts", 11.5, color="muted")
    c.line(300, ly + 24, 322, ly + 24, color="ink", width=1.5, arrow=True)
    c.text(330, ly + 28, "each phase's output file is the next phase's input", 11.5, color="muted")
    return c.svg()


def evaluation(theme):
    c = Canvas(1120, 350, theme, "Five-fold split over perturbation identities, with scoring done by the harness")
    n, folds, size, gap = 20, 5, 17, 5
    gx, gy = 120, 64
    c.text(24, 34, "Perturbation identities, shuffled once with a fixed seed", 15, weight=650)
    for k in range(n):
        c.text(gx + k * (size + gap) + size / 2, gy - 8, f"{k + 1}", 9, color="faint", anchor="middle")
    c.text(gx + n * (size + gap) + 30, gy - 8, "controls", 9.5, color="faint", anchor="middle")
    for f in range(folds):
        y = gy + f * (size + 9)
        c.text(gx - 14, y + 12.5, f"fold {f}", 11.5, color="muted", anchor="end", mono=True)
        test = {k for k in range(n) if k % folds == f}
        rest = [k for k in range(n) if k not in test]
        val = set(rest[:2])
        for k in range(n):
            if k in test:
                c.rect(gx + k * (size + gap), y, size, size, fill="accent", stroke="accent", r=2.5)
            elif k in val:
                c.rect(gx + k * (size + gap) + 0.75, y + 0.75, size - 1.5, size - 1.5, fill="none", stroke="accent",
                       r=2.5, width=1.5)
            else:
                c.rect(gx + k * (size + gap), y, size, size, fill="train", stroke="train", r=2.5)
        c.rect(gx + n * (size + gap) + 8, y, 44, size, fill="control", stroke="control", r=2.5)
    ly = gy + folds * (size + 9) + 22
    items = [("accent", "accent", "test, never trained on", 0), ("none", "accent", "validation", 162),
             ("train", "train", "training", 262), ("control", "control", "controls, always training", 350)]
    for fill, stroke, label, dx in items:
        c.rect(gx + dx, ly - 11, 13, 13, fill=fill, stroke=stroke, r=2, width=1.5 if fill == "none" else 1)
        c.text(gx + dx + 20, ly, label, 11.5, color="muted")
    c.text(gx, ly + 34, "Each run holds out one fold. Across five runs every perturbation is tested exactly once,", 11.5,
           color="muted")
    c.text(gx, ly + 51, "and scores are averaged per perturbation, then across folds (paper, Methods).", 11.5,
           color="muted")

    x, W = 720, 376
    c.text(x, 34, "What each side sees", 15, weight=650)
    c.box(x, 56, W, 50, "Generated script", "train.h5ad, val.h5ad and the test perturbation names")
    c.line(x + W / 2, 106, x + W / 2, 126, arrow=True)
    c.box(x, 130, W, 46, "predictions.h5ad", "one mean log1p(CP10k) profile per test perturbation", title_mono=True)
    c.line(x + W / 2, 176, x + W / 2, 196, arrow=True)
    c.rect(x, 200, W, 120, fill="accent_soft", stroke="accent")
    c.text(x + 14, 222, "Harness scorer", 12.5, weight=600)
    c.text(x + W - 14, 222, "holds test.h5ad", 11, color="muted", anchor="end")
    rows = [
        ("all genes", "MSE, PCC, R², PCC on the change from control"),
        ("DE genes", "top min(50, p/2) by |t| vs control: MSE, PCC, R²"),
        ("ranking", "Recall@50 and AUROC of |predicted change|"),
        ("specific", "PCC from the mean perturbation, discrimination"),
        ("baselines", "no change, mean training shift, linear model"),
    ]
    for i, (k, v) in enumerate(rows):
        c.text(x + 14, 244 + i * 16, k, 11, color="ink", weight=600, mono=True)
        c.text(x + 94, 244 + i * 16, v, 11, color="muted")
    return c.svg()


def failure_modes(theme):
    rows = [("Computation execution error", 41), ("Invalid type or unsupported operation", 23),
            ("Error recovery failure", 16), ("Model configuration error", 6), ("Data access failure", 5),
            ("Other system-level failure", 5), ("Hallucinated structures or outputs", 4)]
    c = Canvas(760, 300, theme, "Failure modes in a manual review of 20 failed CellForge runs, from the paper's Table 1")
    t = c.t
    x0, x1, y0, bar, step = 262, 700, 56, 14, 30
    scale = (x1 - x0) / 50
    c.text(24, 30, "Why runs failed", 15, weight=650)
    c.text(24, 47, "Share of 20 manually reviewed failed runs (paper, Supplement A.2, Table 1)", 11.5, color="muted")
    for v in range(0, 51, 10):
        gx = x0 + v * scale
        c.line(gx, y0 + 6, gx, y0 + 12 + len(rows) * step, color="line", width=1)
        c.text(gx, y0 + 30 + len(rows) * step, f"{v}%", 10.5, color="faint", anchor="middle")
    for i, (label, v) in enumerate(rows):
        y = y0 + 14 + i * step
        c.text(x0 - 12, y + bar - 3, label, 12, color="ink", anchor="end")
        w = v * scale
        c.add(f'<path d="M{x0},{y} H{x0 + w - 4} Q{x0 + w},{y} {x0 + w},{y + 4} V{y + bar - 4} '
              f'Q{x0 + w},{y + bar} {x0 + w - 4},{y + bar} H{x0} Z" fill="{t["accent"]}"/>')
        c.text(x0 + w + 6, y + bar - 2.5, f"{v}%", 11.5, color="ink", weight=600)
    c.line(x0, y0 + 6, x0, y0 + 12 + len(rows) * step, color="faint", width=1)
    return c.svg()


def coordination(theme):
    c = Canvas(640, 330, theme, "Coordination score over rounds when every expert receives the same review score")
    t = c.t
    x0, x1, y0, y1 = 70, 600, 60, 270
    rounds = 6
    lo, hi = 0.45, 1.0

    def X(r):
        return x0 + (x1 - x0) * r / rounds

    def Y(v):
        return y1 - (y1 - y0) * (v - lo) / (hi - lo)

    c.text(24, 28, "How the stop rule behaves", 15, weight=650)
    c.text(24, 45, "c starts at 0.5 and moves 70% of the way to the review score each round", 11.5, color="muted")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        c.line(x0, Y(v), x1, Y(v), color="line", width=1)
        c.text(x0 - 10, Y(v) + 4, f"{v:.1f}", 10.5, color="faint", anchor="end")
    for r in range(rounds + 1):
        c.text(X(r), y1 + 18, str(r), 10.5, color="faint", anchor="middle")
    c.text((x0 + x1) / 2, y1 + 38, "discussion round", 11, color="muted", anchor="middle")
    c.line(x0, Y(0.8), x1, Y(0.8), color="ink", width=1, dash="5 4")
    c.text(x1 - 4, Y(0.8) + 14, "threshold 0.8", 10.5, color="muted", anchor="end")
    for color, review in zip(t["ramp"], (1.0, 0.85, 0.75), strict=True):
        values, score = [0.5], 0.5
        for _ in range(rounds):
            score = 0.3 * score + 0.7 * review
            values.append(score)
        stop = next((r for r in range(1, rounds + 1) if values[r] >= 0.8), None)
        last = stop or rounds
        pts = " ".join(f"{X(r):.1f},{Y(v):.1f}" for r, v in enumerate(values[: last + 1]))
        c.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for r, v in enumerate(values[: last + 1]):
            c.add(f'<circle cx="{X(r):.1f}" cy="{Y(v):.1f}" r="4" fill="{color}" stroke="{t["panel"] if theme == "dark" else "#ffffff"}" stroke-width="2"/>')
        if stop:
            lift = 14 if review == 1.0 else 9
            c.text(X(last) + 8, Y(values[last]) - lift, f"reviews {review:.2f}: stops after round {stop}", 11.5)
        else:
            c.text(X(last), Y(values[last]) + 22, f"reviews {review:.2f}: never reaches 0.8, runs to the cap",
                   11.5, anchor="end")
    return c.svg()


def _mix(a, b, f):
    """Blend two #rrggbb colors, f = 0 gives a and f = 1 gives b."""

    ca, cb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(ca, cb, strict=True))


def _luminance(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable(fill, *inks):
    """The ink with the larger luminance difference from the fill."""

    return max(inks, key=lambda ink: abs(_luminance(ink) - _luminance(fill)))


def ranks(theme):
    groups = [("", ["CellForge"]), ("simple controls", ["Linear", "RF", "Unperturbed"]),
              ("published models", ["CellFlow", "STATE", "GEARS", "scGPT", "Biolord", "CPA", "ChemCPA"]),
              ("generic deep models", ["GNN", "Transformer"])]
    n_rows = sum(len(m) for _, m in groups)
    cw, ch, gap, x0, y0 = 74, 26, 22, 150, 118
    width = x0 + 7 * cw + gap + 24
    c = Canvas(width, y0 + n_rows * ch + 3 * 18 + 34, theme,
               "Overall rank of CellForge and every reference method on seven held-out readouts, from the paper's "
               "Figures 3g and 4g")
    t = c.t
    c.text(24, 30, "Overall rank on held-out perturbations", 15, weight=650)
    c.text(24, 47, "Aggregate of eight metrics, 1 is best, blank where a method was not run (paper, Fig. 3g and 4g)",
           11.5, color="muted")

    def col_x(j):
        return x0 + j * cw + (gap if j >= 3 else 0)

    for j0, j1, label in ((0, 3, "established benchmarks"), (3, 7, "tasks with no published predictor")):
        c.text((col_x(j0) + col_x(j1 - 1) + cw) / 2, 74, label, 11, color="faint", anchor="middle", italic=True)
        c.line(col_x(j0) + 4, 80, col_x(j1 - 1) + cw - 4, 80, color="line", width=1)
    for j, (name, readout) in enumerate(READOUTS):
        c.text(col_x(j) + cw / 2, 96, name, 11.5, weight=600, anchor="middle")
        c.text(col_x(j) + cw / 2, 110, readout, 10.5, color="muted", anchor="middle")
    worst = [max(OVERALL_RANK[m][j] or 0 for m in OVERALL_RANK) for j in range(7)]
    y = y0
    for label, methods in groups:
        if label:
            y += 18
            c.text(24, y - 5, label, 10.5, color="faint", italic=True)
        for m in methods:
            is_us = m == "CellForge"
            c.text(x0 - 12, y + ch / 2 + 4.5, m, 12, color="accent" if is_us else "ink", weight=650 if is_us else 400,
                   anchor="end")
            for j, rank in enumerate(OVERALL_RANK[m]):
                x = col_x(j)
                if rank is None:
                    c.text(x + cw / 2, y + ch / 2 + 4, "·", 12, color="faint", anchor="middle")
                    continue
                f = (rank - 1) / (worst[j] - 1)
                fill = _mix(t["accent"], t["low"], f ** 0.8)
                c.rect(x + 2, y + 2, cw - 4, ch - 4, fill=fill, stroke=fill, r=4)
                c.text(x + cw / 2, y + ch / 2 + 4.5, str(rank), 12, weight=650 if rank == 1 else 400,
                       color=_readable(fill, t["ink"], t["on_dark"]), anchor="middle")
            y += ch
    return c.svg()


def auroc_margin(theme):
    c = Canvas(760, 372, theme, "CellForge AUROC of DE-gene recovery minus the best other method on each readout, "
                                "from the paper's Figures 3f and 4f")
    t = c.t
    x_mid, scale, y0, bar, step = 300, 9000, 92, 16, 30
    value_x = x_mid + 0.015 * scale + 22
    c.text(24, 30, "How far CellForge leads or trails on DE-gene recovery", 15, weight=650)
    c.text(24, 47, "AUROC of CellForge minus the best other method on the same readout (paper, Fig. 3f and 4f)",
           11.5, color="muted")
    rows = []
    for j, (name, readout) in enumerate(READOUTS):
        others = {m: v[j] for m, v in AUROC.items() if m != "CellForge" and v[j] is not None}
        best = max(others.values())
        rival = " and ".join(m for m, v in others.items() if v == best)
        rows.append((f"{name} {readout}", AUROC["CellForge"][j] - best, rival, best))
    top, bottom = y0 - 14, y0 + len(rows) * step
    for v in (-0.01, 0.0, 0.01):
        x = x_mid + v * scale
        c.line(x, top, x, bottom, color="faint" if v == 0 else "line", width=1)
        c.text(x, bottom + 16, f"{v:+.2f}" if v else "0", 10.5, color="faint", anchor="middle")
    c.text(x_mid - 8, top - 4, "trails", 10.5, color="faint", anchor="end", italic=True)
    c.text(x_mid + 8, top - 4, "leads", 10.5, color="faint", italic=True)
    c.text(value_x + 48, top - 4, "closest other method", 10.5, color="faint", anchor="end", italic=True)
    for i, (label, diff, rival, best) in enumerate(rows):
        y = y0 + i * step
        c.text(x_mid - 0.012 * scale - 14, y + bar - 3, label, 12, anchor="end")
        w = abs(diff) * scale
        x = x_mid if diff >= 0 else x_mid - w
        c.add(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar}" rx="3" '
              f'fill="{t["accent"] if diff >= 0 else t["behind"]}"/>')
        c.text(value_x + 48, y + bar - 3, f"{diff:+.3f}", 11.5, weight=600, anchor="end")
        c.text(value_x + 60, y + bar - 3, f"{rival}, {best:.3f}", 11.5, color="muted")
    c.text(24, bottom + 48, "Every gap is within 0.015 AUROC, and a simple control (linear, random forest or", 11.5,
           color="muted")
    c.text(24, bottom + 65, "unperturbed) is the closest other method on six of the seven readouts.", 11.5,
           color="muted")
    return c.svg()


def mark(theme):
    t = THEMES[theme]
    pts = [(60 + 23 * math.cos(math.radians(a)), 60 + 23 * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    edges = "".join(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'
                    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True))
    nodes = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8.5" fill="{color}"/>' for (x, y), color in zip(pts, t["phase"], strict=True))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120" role="img" '
            f'aria-label="CellForge mark: three agents inside a cell membrane">'
            f'<g fill="none" stroke="{t["muted"]}" stroke-width="2.5"><circle cx="60" cy="60" r="54"/>'
            f'<circle cx="60" cy="60" r="47.5" stroke-dasharray="2.5 4.2"/></g>'
            f'<g stroke="{t["faint"]}" stroke-width="2.25">{edges}</g>{nodes}</svg>\n')


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "figs"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pipeline", pipeline), ("evaluation", evaluation), ("failure-modes", failure_modes),
                     ("coordination", coordination), ("mark", mark), ("ranks", ranks),
                     ("auroc-margin", auroc_margin)):
        for theme in THEMES:
            suffix = "" if theme == "light" else "-dark"
            (out / f"{name}{suffix}.svg").write_text(fn(theme), encoding="utf-8")


if __name__ == "__main__":
    main()
