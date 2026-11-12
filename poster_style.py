#!/usr/bin/env python
"""
poster_style.py -- one import that makes every figure legible at poster size
=============================================================================

WHY THIS EXISTS INSTEAD OF EDITING EVERY fontsize= BY HAND
    The plotting scripts carry ~40 hand-tuned `fontsize=` arguments between
    them, and those arguments OVERRIDE rcParams. So bumping rcParams alone
    changes nothing on the text that matters -- axis labels, panel titles,
    annotations -- and hand-editing every one of them means the relative
    hierarchy (title > label > tick > caption) has to be re-derived by eye
    each time we change our minds about the size.

    This module scales whatever number the script asked for. The hierarchy the
    scripts already encode is preserved exactly; the whole thing just gets
    bigger. Change SCALE in one place and every figure moves together.

>>> THE TARGET, IN PRINTED POINTS
    The poster is 182.88 x 106.68 cm (72 x 42 in) in four columns of
    0.233\\textwidth, so a column is about 16 in wide. Body text is beamer 11 pt
    at beamerposter scale=1.62, i.e. ~17.8 pt as printed. A figure goes in at
    width=1.0\\textwidth, so what a viewer actually reads is

        printed_pt = source_pt x SCALE x (16 in / figure_width_in)

    At the old settings that is 11 x 1.0 x (16/13.2) = 13.3 pt, which is 75% of
    the body copy beside it. That is why the figures read as small on
    poster_v3.png even though nothing is wrong with them. The target is at or
    just above body copy: a viewer standing back should be able to read the
    figure before the paragraph.

>>> WHY THE CANVAS GROWS TOO, AND WHY THAT IS NOT SELF-DEFEATING
    Text and canvas are two separate knobs and they do different jobs:

        SCALE / GROW   sets how CROWDED the figure is (text vs plot area)
        SCALE / GROW x 16/w_orig   sets how BIG the text is when printed

    Growing the canvas alone would shrink the printed type, since the figure is
    scaled down to the column either way. Growing BOTH buys elbow room for the
    hand-placed annotations at no cost in printed size, and costs nothing on
    the poster because a uniform grow leaves the aspect ratio -- and therefore
    the vertical space the figure eats -- unchanged.

        GROW 1.25, SCALE 1.75  ->  crowding 1.40x, printed ~19 pt

    1.40x crowding was verified by eye on falsify_2panel; 1.55x collided.

>>> THE ONE THING THAT CAN GO WRONG
    Bigger text in a fixed-size axes collides. Every annotation in these
    scripts was placed against the OLD metrics. After applying this, LOOK at
    each figure. Legends outgrowing their axes and arrows landing on top of
    labels are the two failure modes, and neither raises an exception.

USAGE
    import poster_style; poster_style.apply()          # at the top of a script
    fig = plt.figure(figsize=poster_style.figsize(13.0, 5.4))

    POSTER_SCALE=1.0 POSTER_GROW=1.0 python plot_whatever.py   # the old look
    POSTER_SCALE=2.0 python plot_whatever.py                   # louder
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

SCALE = float(os.environ.get("POSTER_SCALE", 1.75))
BOLD = os.environ.get("POSTER_BOLD", "1") != "0"

_PATCHED = False


def _patch_text_sizes(scale):
    """Scale every explicitly requested font size.

    Text.set_size is the single funnel: set_fontsize is an alias for it, and
    the Text constructor routes `fontsize=` through it too. Patching here
    catches axis labels, titles, annotations, legend entries and tick labels
    in one place.

    >>> IDEMPOTENCE. Matplotlib re-sets sizes internally (legends copy font
    properties, tight_layout re-measures). Multiplying blindly would compound
    on every one of those calls and produce 40 pt tick labels. So each Text
    remembers the raw value it was last given; a repeat of that same raw value
    is re-scaled rather than scaled again.

    >>> WHY THIS WRITES TO _fontproperties INSTEAD OF CALLING THE ORIGINAL
    (fixed 2026-09-07)
    set_size and set_fontsize are not two methods -- one is a generated alias
    that forwards to the other, and which one is canonical is a matplotlib
    implementation detail. The first version captured `Text.set_size`, patched
    BOTH names, then called the captured function; on this matplotlib the
    captured one was the alias, so it forwarded straight back into the patch:

        RecursionError: maximum recursion depth exceeded

    Writing to the FontProperties object directly is what the real setter does
    anyway, and it cannot round-trip through an alias.
    """

    def set_size(self, size):
        if isinstance(size, (int, float)) and not isinstance(size, bool):
            raw = getattr(self, "_poster_raw_size", None)
            if raw is not None and abs(size - raw * scale) < 1e-9:
                size = raw * scale          # re-set of an already-scaled value
            else:
                self._poster_raw_size = float(size)
                size = float(size) * scale
        self._fontproperties.set_size(size)   # what Text.set_fontsize does
        self.stale = True

    Text.set_size = set_size
    Text.set_fontsize = set_size


def apply(scale=None, bold=None):
    """Install the poster look. Safe to call more than once."""
    global _PATCHED
    s = SCALE if scale is None else float(scale)
    b = BOLD if bold is None else bool(bold)

    weight = "bold" if b else "normal"
    plt.rcParams.update({
        # >>> BASE SIZES, NOT SCALED ONES (fixed 2026-09-07).
        # These must NOT be pre-multiplied by s. Matplotlib builds a Text with
        # the rcParam value and then calls set_size on it, which goes through
        # the patch below -- so a scaled rcParam gets scaled a SECOND time.
        # The symptom was subtle: figures whose scripts set every fontsize
        # explicitly looked right, while make_comparison_figure.py, which
        # relies on the defaults for its axis labels, came out with axis labels
        # at 11 x 1.75^2 = 34 pt, twice the size of its own tick labels.
        # These numbers are the matplotlib defaults this project was written
        # against; the patch is the single place scaling happens.
        "font.size":         11.0,
        "axes.titlesize":    12.5,
        "axes.labelsize":    11.0,
        "xtick.labelsize":   10.0,
        "ytick.labelsize":   10.0,
        "legend.fontsize":    9.5,
        "figure.titlesize":  14.5,
        # weight. rcParams is what unmarked text inherits, so tick labels and
        # legend entries go bold here without touching the scripts.
        "font.weight":        weight,
        "axes.labelweight":   weight,
        "axes.titleweight":   weight,
        "figure.titleweight": weight,
        # heavier furniture, so the type does not overpower hairline axes
        "axes.linewidth":     1.0 * max(1.0, s * 0.9),
        "xtick.major.width":  1.0 * max(1.0, s * 0.9),
        "ytick.major.width":  1.0 * max(1.0, s * 0.9),
        "xtick.major.size":   4.0 * max(1.0, s * 0.9),
        "ytick.major.size":   4.0 * max(1.0, s * 0.9),
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "savefig.dpi":        300,
    })

    if not _PATCHED:
        _patch_text_sizes(s)
        _PATCHED = True
    return s


def figsize(w, h, grow=None):
    """Optionally enlarge the canvas along with the type.

    Growing the FIGURE as well as the font keeps the plot area from being
    eaten by the bigger labels. Pass grow=1.0 to keep the canvas fixed, which
    is what you want when the poster column width is the real constraint and
    the figure is going in at width=1.0\\textwidth regardless.
    """
    g = float(os.environ.get("POSTER_GROW", 1.25)) if grow is None else float(grow)
    return (w * g, h * g)


def line(w):
    """Scale a hand-set linewidth by the canvas growth.

    A `lw=2.8` written against a 13 in canvas is a thinner-looking line on a
    16 in canvas. Curves and arrows that carry meaning should hold their
    apparent weight; grid lines and hairlines can be left alone.
    """
    return w * float(os.environ.get("POSTER_GROW", 1.25))
