"""Reliability diagrams as self-contained SVG. `docs/07`: "Publish these."

    svg = reliability_diagram(ece.equal_mass, title="Cournot Prior — published")
    Path("reliability.svg").write_text(svg)

No plotting dependency, matching `cournot.metrics`, which is pure Python for the same
reason: these inputs are eval-sized and a readable artifact is worth more than
speed. The output is a deterministic string, which is also what makes it testable.

**The diagram shows bin counts, and that is not decoration.** A bin holding three
questions plots identically to one holding three thousand, and the eye reads a
wobble in a three-question bin as miscalibration. `cournot.metrics` already tracks
`smallest_populated_bin` because of this; a diagram that drops the counts throws
away the warning. Marker area is proportional to `n`, and a count strip under the
plot gives the numbers outright.

**Empty bins stay empty.** `Bin` retains them with `n == 0` precisely so a
diagram can show where there was no data, and the polyline breaks across them
rather than interpolating a segment nobody measured.

Theme-aware: the palette is `references/palette.md`'s validated steps, with a
`prefers-color-scheme` block. One data series, so no legend — the title names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

from cournot.metrics import Bin, ECEBinned

__all__ = ["DiagramGeometry", "reliability_diagram"]


@dataclass(frozen=True)
class DiagramGeometry:
    """Layout in user units. Defaults suit a model card at ~2x."""

    width: int = 460
    height: int = 420
    pad_left: int = 52
    pad_right: int = 16
    pad_top: int = 44
    pad_bottom: int = 30
    strip_height: int = 44
    """Height of the count strip beneath the plot. It shares the x axis with the
    plot above rather than adding a second y scale to it — two panels, never two
    scales on one."""

    strip_gap: int = 34
    """Vertical space between the plot baseline and the count strip. Must clear
    the x-axis tick labels, which sit 14px below the baseline; at 10 the strip's
    caption landed on top of them."""

    @property
    def plot_height(self) -> int:
        return self.height - self.pad_top - self.pad_bottom - self.strip_height - self.strip_gap

    @property
    def plot_width(self) -> int:
        return self.width - self.pad_left - self.pad_right


_STYLE = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--series:#2a78d6}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#ffffff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5}}
.bg{fill:var(--surface)}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ideal{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3;fill:none}
.series{stroke:var(--series);stroke-width:2;fill:none}
.pt{fill:var(--series);stroke:var(--surface);stroke-width:2}
.bar{fill:var(--series);opacity:.5}
.t{fill:var(--ink);font:600 13px system-ui,sans-serif}
.s{fill:var(--ink2);font:11px system-ui,sans-serif}
.m{fill:var(--muted);font:10px system-ui,sans-serif}
""".strip()


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def reliability_diagram(
    binned: ECEBinned,
    *,
    title: str,
    subtitle: str = "",
    geometry: DiagramGeometry | None = None,
) -> str:
    """Render one binned ECE result as a standalone SVG document."""
    geo = geometry or DiagramGeometry()
    bins: tuple[Bin, ...] = binned.bins
    if not bins:
        raise ValueError("cannot draw a reliability diagram with no bins")

    x0, y0 = geo.pad_left, geo.pad_top
    w, h = geo.plot_width, geo.plot_height

    def px(p: float) -> float:
        return x0 + p * w

    def py(p: float) -> float:
        return y0 + (1.0 - p) * h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {geo.width} {geo.height}" '
        f'width="{geo.width}" height="{geo.height}" role="img" '
        # quoteattr, not escape: escape() leaves quotes alone, and a title
        # containing one would close the attribute and break the document.
        f"aria-label={quoteattr(title)}>",
        f"<style>{_STYLE}</style>",
        f'<rect class="bg" x="0" y="0" width="{geo.width}" height="{geo.height}"/>',
        f'<text class="t" x="{geo.pad_left}" y="20">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(f'<text class="s" x="{geo.pad_left}" y="36">{escape(subtitle)}</text>')

    # Recessive grid and axis labels at the quartiles.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line class="grid" x1="{x0}" y1="{py(frac):.1f}" '
            f'x2="{px(1.0):.1f}" y2="{py(frac):.1f}"/>'
        )
        parts.append(
            f'<text class="m" x="{x0 - 6}" y="{py(frac) + 3:.1f}" text-anchor="end">{frac:g}</text>'
        )
        parts.append(
            f'<text class="m" x="{px(frac):.1f}" y="{y0 + h + 14:.1f}" '
            f'text-anchor="middle">{frac:g}</text>'
        )

    # Perfect calibration.
    parts.append(
        f'<line class="ideal" x1="{px(0.0):.1f}" y1="{py(0.0):.1f}" '
        f'x2="{px(1.0):.1f}" y2="{py(1.0):.1f}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{x0}" y1="{y0 + h:.1f}" x2="{px(1.0):.1f}" y2="{y0 + h:.1f}"/>'
    )

    # The series. Runs break at empty bins rather than spanning them.
    runs: list[list[Bin]] = []
    current: list[Bin] = []
    for b in bins:
        if b.n > 0 and b.mean_forecast is not None and b.observed_frequency is not None:
            current.append(b)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    for run in runs:
        if len(run) > 1:
            pts = " ".join(
                f"{px(b.mean_forecast):.1f},{py(b.observed_frequency):.1f}"  # type: ignore[arg-type]
                for b in run
            )
            parts.append(f'<polyline class="series" points="{pts}"/>')

    largest = max((b.n for b in bins), default=1) or 1
    for b in bins:
        if b.n == 0 or b.mean_forecast is None or b.observed_frequency is None:
            continue
        # Area proportional to n, so a thin bin cannot masquerade as a thick one.
        radius = 3.0 + 6.0 * (b.n / largest) ** 0.5
        parts.append(
            f'<circle class="pt" cx="{px(b.mean_forecast):.1f}" '
            f'cy="{py(b.observed_frequency):.1f}" r="{radius:.1f}">'
            f"<title>forecast {_fmt(b.mean_forecast)}, observed "
            f"{_fmt(b.observed_frequency)}, n={b.n}</title></circle>"
        )

    # Count strip: the numbers outright, sharing the x axis above.
    strip_top = y0 + h + geo.strip_gap
    parts.append(
        f'<text class="m" x="{x0}" y="{strip_top - 4:.1f}">questions per bin '
        f"(min {binned.smallest_populated_bin}, {binned.n_populated_bins} populated)</text>"
    )
    bar_w = w / len(bins)
    for i, b in enumerate(bins):
        if b.n == 0:
            continue
        bar_h = (geo.strip_height - 14) * (b.n / largest)
        bx = x0 + i * bar_w
        parts.append(
            f'<rect class="bar" x="{bx + 1:.1f}" '
            f'y="{strip_top + (geo.strip_height - 14) - bar_h:.1f}" '
            f'width="{max(bar_w - 2, 1):.1f}" height="{bar_h:.1f}" rx="2">'
            f"<title>bin [{_fmt(b.lower)}, {_fmt(b.upper)}): n={b.n}</title></rect>"
        )

    parts.append("</svg>")
    return "\n".join(parts)
