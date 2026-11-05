"""Open the grasp visualizations of an eval run in the browser.

visualizations.json (written by eval_egf.py via the reference save_figure) holds one plotly figure
with every test object and a dropdown to switch between them. This wraps it into a standalone HTML
file (plotly.js from the CDN) and opens it; with an object name, only that object is kept, which is
much smaller (~5 MB instead of 150-400 MB) and opens instantly.

    python scripts/show_vis.py results/eval/equigraspflow_full/paper            # all objects, dropdown
    python scripts/show_vis.py results/eval/equigraspflow_full/paper Mug_12     # one object
    python scripts/show_vis.py results/eval/equigraspflow_full/paper --list

Shown per object: the mesh and up to 100 (partial: 25) randomly chosen generated grasps, one
subplot per test rotation (partial: rows = rotations, columns = viewpoints, plus the input points).
"""

import argparse
import json
import webbrowser
from pathlib import Path

from plotly.offline import get_plotlyjs_version

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-{plotlyjs}.min.js"></script></head>
<body style="margin:0"><div id="fig" style="width:100vw;height:100vh"></div>
<script>const fig = {fig}; Plotly.newPlot("fig", fig.data, fig.layout);</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="eval output directory containing visualizations.json")
    parser.add_argument("object", nargs="?", help="<category>_<object index>, e.g. Mug_12 (default: all)")
    parser.add_argument("--list", action="store_true", help="list the available objects and exit")
    args = parser.parse_args()

    fig = json.loads((args.eval_dir / "visualizations.json").read_text())
    buttons = fig["layout"]["updatemenus"][0]["buttons"]
    labels = [b["label"] for b in buttons]
    if args.list:
        print("\n".join(labels))
        return

    if args.object:
        if args.object not in labels:
            raise SystemExit(f"unknown object {args.object!r}; use --list")
        visible = buttons[labels.index(args.object)]["args"][0]["visible"]
        fig["data"] = [dict(t, visible=True) for t, v in zip(fig["data"], visible, strict=True) if v]
        del fig["layout"]["updatemenus"]
    title = args.object or args.eval_dir.as_posix()
    fig["layout"]["title"] = {"text": title}

    out = args.eval_dir / "vis" / f"{args.object or 'all'}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        HTML.format(title=title, plotlyjs=get_plotlyjs_version(), fig=json.dumps(fig)), encoding="utf-8"
    )
    print(out)
    webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
