# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as ticker

# %%


def id_generator(df, hole_col, from_col, to_col, composite=False):
    """Generate sample names from hole_id and depth
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    hole_col : str
        column name of hole_id
    from_col : str
        column name of start depth
    to_col : str
        column name of end depth
    composite : bool, optional, default False
        whether to return a composite id name in the form
        'hole_id (from-to)' or to return two seperate
        names in the form 'hole_id from' and 'hole_id to'
    Returns
    -------
    pandas Series
        1 or 2 pandas Series of type str of same length as original data
        with generated id names
    """
    if composite:
        return df.apply(lambda x: f"{x[hole_col]} ({x[from_col]}-{x[to_col]})", axis=1)
    else:
        return df[hole_col] + " " + df[from_col].astype(str), df[hole_col] + " " + df[
            to_col
        ].astype(str)


def roundup(num, base):
    """Rounds a number UP to the closest value divisible by base
    Parameters
    ----------
    num : numeric
        value to be rounded
    base : int
        base system to round in
    Returns
    -------
    int
    """
    return base * math.ceil(num / base)


def lith_linear_proportion(
    df,
    lith_col,
    interval_col="interval",
    norm_to=100,
    new_name="lith_lin_pct",
    percent=True,
):
    """Calculates linear proportion of each lithology present in the study area
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    lith_col : str
        column name of lithology column
    interval_col : str, optional, default 'interval'
        column name of column with intervals (distance)
    norm_to : numeric, optional, default 100
        value to normalize to, 100 for pct
    new_name : str, optional, default 'lith_lin_pct'
        name of the series returned
    percent : bool, optional, default True
        whether to return a percent value or not
    Returns
    -------
    pandas Series
    """
    if percent:
        temp_series = (
            df.groupby(by=lith_col)[interval_col].sum() / df[interval_col].sum()
        ) * norm_to
    else:
        temp_series = df.groupby(by=lith_col)[interval_col].sum()
    temp_series = temp_series.rename(new_name)
    return temp_series


def pyramid_plotter(df, env_col, exp_col, raw_data=False):
    """Creates a pyramid plot (opposing bar plot) from two columns within a df
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    env_col : str
        name of first column to make bar plot from
    exp_col : str
        name of second column to make bar plot from
    raw_data : bool, optional, default False
        toggle whether to plot percentile or raw data
    Returns
    -------
    matplotlib.pyplot figure
    """
    sns.set_theme(style="darkgrid")
    limx = df.abs().max().max()
    limx_r = roundup(limx, 5)
    if limx_r - limx < 2.5:
        limx_r += 5
    if raw_data:
        limx_r = roundup(limx_r, 100)
    fig1, ax1 = plt.subplots(figsize=(20, 20))
    _ = sns.barplot(
        x=df[env_col], y=df.index, ax=ax1, color="r", label="Environmental", lw=0
    )
    _ = sns.barplot(
        x=df[exp_col], y=df.index, ax=ax1, color="b", label="Exploration", lw=0
    )
    ax1.tick_params(axis="y", which="major", labelsize=20)
    ax1.set_xlabel("Percentage of Total Sampled Footage", fontsize=30)
    # the 'raw_data' offers functionality to plot
    # the raw distance, as opposed to normalized distance,
    # of core samples, it is clunky, but works
    # anywhere 'raw_data' shows up know that this it its purpose
    if raw_data:
        ax1.set_xlabel("Total Sampled Footage")
    ax1.set_ylabel("Lithology", fontsize=30)
    ax1.set_title("Environmental vs Exploration", fontsize=35)
    x_range = list(np.arange(-limx_r, limx_r + 1, 5))
    x_range.pop(0)
    x_range.pop(-1)
    if raw_data:
        x_range = list(np.linspace(-limx_r, limx_r, 5).round())
    ax1.xaxis.set_ticks(x_range)  # type: ignore
    xticklabels = [f"{x}%" for x in np.abs(x_range)]
    if raw_data:
        xticklabels = [f"{int(x)}" for x in np.abs(x_range)]
    ax1.xaxis.set_ticklabels(xticklabels, fontsize=25)  # type: ignore
    ax1.set_xlim(-limx_r, limx_r)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=h,
        labels=["Environmental", "Exploration"],
        ncol=2,
        loc="center",
        frameon=False,
        fontsize=23,
        bbox_to_anchor=[0.52, -0.09],
    )
    # label each bar with its percentage
    for bar in ax1.patches:
        height = bar.get_width()
        # do not label the 0% values
        if np.isclose(height, 0.0):
            continue
        lab = " {:.2f}% ".format(np.abs(height))
        if raw_data:
            lab = "{}".format(int(np.abs(height)))
        if height < 0:
            if np.abs(height) > (limx_r / 2):
                ax1.text(
                    x=height,
                    y=bar.get_y() + bar.get_height() / 2,
                    s=lab,
                    fontsize=17,
                    color="w",
                    ha="left",
                    va="center",
                )
            else:
                ax1.text(
                    x=height,
                    y=bar.get_y() + bar.get_height() / 2,
                    s=lab,
                    fontsize=17,
                    color="k",
                    ha="right",
                    va="center",
                )
        else:
            if np.abs(height) > (limx_r / 2):
                ax1.text(
                    x=height,
                    y=bar.get_y() + bar.get_height() / 2,
                    s=lab,
                    fontsize=17,
                    color="w",
                    ha="right",
                    va="center",
                )
            else:
                ax1.text(
                    x=height,
                    y=bar.get_y() + bar.get_height() / 2,
                    s=lab,
                    fontsize=17,
                    color="k",
                    ha="left",
                    va="center",
                )

    return fig1


def bplot_lith_prop(
    df,
    lithorder,
    xlab,
    title,
    proprt,
    group_by="lithology_relog",
    lines=False,
    lineloc=(0.2, 2, 3),
    pal="turbo",
    scale_log=False,
):
    """Creates boxplots of sample property grouped by lithology
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    lithorder : listlike
        list of lithologies to plot, in order
    xlab : str
        label for the x-axis
    title : str
        title for the plot, put None for no title
    proprt : str
        column name of property to make boxplot of
    group_by : str, optional, default 'lithology_relog'
        property to group by
    lines : bool, optional, default False
        whether to plot vertical lines
    lineloc : listlike, optional, default (0.2,2,3)
        x-axis locations for vertical lines
    pal : palette-like, optional, default 'turbo'
        either named seaborn palette or custom palette
    scale_log : bool, optional, default False
        whether to plot x-axis with log scale
    Returns
    -------
    fig : matplotlib Figure
        Figure containing the plot
    """
    sns.set_theme(style="darkgrid")
    # filter lithology to those present in the study area
    lithin = df[group_by].unique()
    lithorder = [lith for lith in lithorder if lith in lithin]
    # highest value of the target variable
    high = df[proprt].max()
    low = df[proprt].min()
    figx, axx = plt.subplots(figsize=(20, 20))
    _ = sns.boxplot(data=df, y=group_by, x=proprt, order=lithorder, palette=pal, ax=axx)
    if scale_log:
        axx.set_xlim([min(-0.5, low - 0.8), high])  # type: ignore
        axx.set_xscale("symlog")
        axx.xaxis.set_major_formatter(ticker.ScalarFormatter())
        lab_x = min(-0.2, low - 0.4)  # lab_x for plotting anotations
    else:
        a, b = axx.get_xlim()
        x_range = b - a
        # empirical ratios that make good placement for the labeling
        axx.set_xlim(a - x_range / 15, b)
        lab_x = a - (x_range / 15) * 0.2  # lab_x for plotting anotations
    axx.set_xlabel(xlab, fontsize=25)
    axx.set_ylabel(None)  # type: ignore
    y_max, y_min = axx.get_ylim()
    axx.set_ylim(y_max + 0.5, y_min - 0.5)
    axx.tick_params(axis="both", which="major", labelsize=25)
    if lines:
        axx.vlines(
            lineloc,
            ymin=0,
            ymax=1,
            colors="#980043",  # type: ignore
            linestyles="--",  # type: ignore
            transform=axx.get_xaxis_transform(),
        )
    axx.set_title(title, fontsize=30)
    cnts = df.groupby(by=group_by, sort=False)[group_by].count()
    cnts = cnts.reindex(lithorder)
    cnt_lab = [f"n: {cnt}" for cnt in cnts]

    for lab, loca in zip(cnt_lab, axx.get_yticks()):
        axx.text(
            lab_x,
            loca,
            lab,
            horizontalalignment="center",
            verticalalignment="center",
            size=20,
            color="k",
        )

    return figx
