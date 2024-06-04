class PlotStyle:
    ci_line = dict(color="magenta", lw=1.5, ls="--", alpha=0.8, zorder=4.0)
    switch_line = dict(color="tab:brown", lw=1.0, alpha=0.5, zorder=5.5)
    obs_region = dict(facecolor="0.6", edgecolor="none", alpha=0.4, zorder=3, hatch="/")
    goal_region = dict(facecolor="green", edgecolor="none", alpha=0.3, zorder=4.0)

    valid_color = "lime"
    invalid_color = "#c1272d"

    ObsColor = "#988ED5"
    ZeroColor = "magenta"
    BoundColor = "#FBC15E"
