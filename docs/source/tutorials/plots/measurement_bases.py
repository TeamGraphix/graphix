"""Bloch-sphere illustration of the XY / XZ / YZ measurement planes (shared camera)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "white",
})
INK, MUTED, OUTLINE = "#1F2937", "#9CA3AF", "#CBD5E1"

E = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
ALPHA = np.deg2rad(50)                       # illustrative value of alpha
ALPHA_TAGS = ["0", r"\pi/2", r"\pi", r"3\pi/2"]
CAMERA = (40, 25)                            # (azimuth, elevation) in degrees, shared by all panels

# ref: axis where alpha = 0 ; perp: axis where alpha = pi/2
PLANES = [
    dict(name="XY", ref="X", perp="Y", color="#6366F1",
         state=r"$|\pm_{\mathrm{XY},\alpha}\rangle=\frac{1}{\sqrt{2}}"
               r"\left(|0\rangle\pm e^{i\alpha}|1\rangle\right)$"),
    dict(name="XZ", ref="Z", perp="X", color="#0D9488",
         state=r"$|\pm_{\mathrm{XZ},\alpha}\rangle=t^{\alpha}_{\pm}|0\rangle"
               r"\pm t^{\alpha}_{\mp}|1\rangle$"),
    dict(name="YZ", ref="Z", perp="Y", color="#E11D48",
         state=r"$|\pm_{\mathrm{YZ},\alpha}\rangle=t^{\alpha}_{\pm}|0\rangle"
               r"\pm i\,t^{\alpha}_{\mp}|1\rangle$"),
]


# --------------------------------------------------------------------------- #
# Tiny orthographic camera
# --------------------------------------------------------------------------- #
class View:
    def __init__(self, az_deg, el_deg):
        az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
        self.c = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.r = np.array([-np.sin(az), np.cos(az), 0.0])
        self.u = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])

    def xy(self, p):
        p = np.asarray(p)
        return np.stack([p @ self.r, p @ self.u], axis=-1)

    def depth(self, p):                      # > 0: towards the viewer
        return np.asarray(p) @ self.c


VIEW = View(*CAMERA)                         # one camera for the whole figure


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_sphere(ax):
    n = 500
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    rr = np.hypot(X, Y)
    t = np.clip(np.hypot(X + 0.35, Y - 0.40) / 1.7, 0, 1) ** 1.2      # soft highlight
    light, dark = np.array([0.975, 0.98, 1.0]), np.array([0.80, 0.84, 0.90])
    rgb = light + (dark - light) * t[..., None]
    alpha = np.clip((1 - rr) * n / 2, 0, 1)                            # anti-aliased edge
    ax.imshow(np.dstack([rgb, alpha]), extent=(-1, 1, -1, 1), origin="lower",
              zorder=0, interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=OUTLINE, lw=1.0, zorder=1))


def great_circle(ax, e1, e2, color, lw, a_front=1.0, a_back=0.45, z=3):
    th = np.linspace(0, 2 * np.pi, 721)
    P = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    xy, front = VIEW.xy(P), VIEW.depth(P) >= 0
    for mask, ls, a, zo in ((front, "-", a_front, z + 3),
                            (~front, (0, (3, 3)), a_back, z)):
        seg = np.where(mask[:, None], xy, np.nan)
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, ls=ls, alpha=a,
                zorder=zo, solid_capstyle="round")


def shade_plane(ax, e1, e2, color):
    """Radially-graded translucent disc lying in the plane spanned by e1, e2."""
    th = np.linspace(0, 2 * np.pi, 361)
    P = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    ax.add_patch(Polygon(VIEW.xy(P), closed=True, fc=color, ec="none", alpha=0.10, zorder=2))
    for s in np.linspace(1, 0.05, 14):
        ax.add_patch(Polygon(VIEW.xy(s * P), closed=True, fc=color, ec="none",
                             alpha=0.024, zorder=2))


def label_at(ax, p3, text, off=0.12, **kw):
    """Place text just outside the screen projection of p3, along the radial direction."""
    q = VIEW.xy(p3)
    n = np.linalg.norm(q)
    d = q / n if n > 0.2 else np.array([0.0, 1.0])
    ha = "left" if d[0] > 0.35 else "right" if d[0] < -0.35 else "center"
    va = "bottom" if d[1] > 0.35 else "top" if d[1] < -0.35 else "center"
    kw.setdefault("alpha", 1.0 if VIEW.depth(p3) >= -0.05 else 0.55)
    ax.text(*(q + off * d), text, ha=ha, va=va, multialignment=ha,
            linespacing=1.15, zorder=12, **kw)


def draw_axes(ax, plane):
    """All six half-axes carry the same label at the same place in every panel."""
    L, col = 1.12, plane["color"]
    in_plane = (plane["ref"], plane["perp"])
    order = [(plane["ref"], +1), (plane["perp"], +1), (plane["ref"], -1), (plane["perp"], -1)]
    tags = dict(zip(order, ALPHA_TAGS))

    for name, vec in E.items():
        for sgn in (+1, -1):
            end = sgn * L * vec
            front = VIEW.depth(end) >= 0
            is_ref = (name == plane["ref"] and sgn > 0)
            c = col if is_ref else (INK if name in in_plane else MUTED)
            lw = 1.8 if is_ref else (1.1 if name in in_plane else 0.9)
            q = VIEW.xy(end)
            ax.plot([0, q[0]], [0, q[1]], color=c, lw=lw,
                    ls="-" if front else (0, (3, 3)),
                    alpha=0.95 if front else 0.5, zorder=5 if front else 1.5,
                    solid_capstyle="round")
            if front and sgn > 0:                       # small arrow head
                ax.annotate("", xy=q, xytext=VIEW.xy(0.85 * end), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                            mutation_scale=8, shrinkA=0, shrinkB=0))

            sign = "+" if sgn > 0 else "\N{MINUS SIGN}"
            p = sgn * vec
            if name in in_plane:
                ax.scatter(*VIEW.xy(p), s=30, color=col, edgecolor="white", linewidth=0.9,
                           zorder=6 if VIEW.depth(p) >= 0 else 3)
                label_at(ax, end,
                         rf"$\mathbf{{{sign}{name}}}$" + "\n" + rf"$\alpha={tags[(name, sgn)]}$",
                         color=col, fontsize=9.5)
            else:
                label_at(ax, end, rf"$\mathbf{{{sign}{name}}}$", color=MUTED, fontsize=9.5)


def draw_measurement(ax, plane):
    col, ref, perp, name = plane["color"], E[plane["ref"]], E[plane["perp"]], plane["name"]
    v = np.cos(ALPHA) * ref + np.sin(ALPHA) * perp
    q = VIEW.xy(v)

    # antipodal (-) outcome
    ax.plot([0, -q[0]], [0, -q[1]], color=col, lw=1.4, ls=(0, (2, 2)), alpha=0.55, zorder=2.5)
    ax.scatter(*(-q), s=48, facecolor="white", edgecolor=col, linewidth=1.6, zorder=4)
    label_at(ax, -v, rf"$|-_{{{name},\alpha}}\rangle$", off=0.10, color=col, fontsize=11)

    # angle wedge + arc
    R = 0.42
    th = np.linspace(0, ALPHA, 60)
    arc = VIEW.xy(R * (np.cos(th)[:, None] * ref + np.sin(th)[:, None] * perp))
    ax.add_patch(Polygon(np.vstack([[0, 0], arc]), closed=True, fc=col, ec="none",
                         alpha=0.28, zorder=7))
    ax.plot(arc[:, 0], arc[:, 1], color=col, lw=1.8, zorder=8, solid_capstyle="round")
    ax.annotate("", xy=arc[-1], xytext=arc[-4], zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    mid = np.cos(ALPHA / 2) * ref + np.sin(ALPHA / 2) * perp
    ax.text(*VIEW.xy(0.68 * mid), r"$\alpha$", color=col, fontsize=14,
            ha="center", va="center", zorder=12)

    # measurement vector (+) outcome
    ax.annotate("", xy=q, xytext=(0, 0), zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                mutation_scale=16, shrinkA=0, shrinkB=0))
    ax.scatter(0, 0, s=22, color=INK, edgecolor="white", linewidth=0.8, zorder=10)
    label_at(ax, v, rf"$|+_{{{name},\alpha}}\rangle$", off=0.1, color=col, fontsize=12, fontweight="bold")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), constrained_layout=True)
    fig.suptitle("Measurement bases", fontsize=19, fontweight="bold",
                     color=INK)
    for ax, plane in zip(axes, PLANES):
        e1, e2 = E[plane["ref"]], E[plane["perp"]]

        draw_sphere(ax)
        # faint reference great circles + highlighted measurement plane
        for a, b in (("X", "Y"), ("X", "Z"), ("Y", "Z")):
            if {a, b} != {plane["ref"], plane["perp"]}:
                great_circle(ax, E[a], E[b], OUTLINE, 0.8, 0.9, 0.5, z=2)
        shade_plane(ax, e1, e2, plane["color"])
        great_circle(ax, e1, e2, plane["color"], 2.0, 1.0, 0.5, z=3)

        draw_axes(ax, plane)
        draw_measurement(ax, plane)

        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{plane['name']} plane", fontsize=15, fontweight="semibold",
                     color=INK, pad=8)
        ax.set_xlabel(plane["state"], fontsize=13,
                      color=INK, labelpad=6, linespacing=1.7)
    return fig


fig = make_figure()                          # module-level, so `.. plot::` picks it up
