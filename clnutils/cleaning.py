# %%
import numpy as np
import pandas as pd
from clnutils import super_sub_scriptreplace

# %%


def overlap(
    df,
    hole_col="drill_hole",
    dfrom="from",
    dto="to",
    samp_id1="sample_id",
    intv="interval",
):
    """Assesses drillhole overlap between samples.
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    hole_col : str, optional, default 'drill_hole'
        column name of hole_id
    dfrom : str, optional, default 'From'
        column name of start depth
    dto : str, optional, default 'To'
        column name of end depth
    samp_id1 : str, optional, default 'Sample ID'
        column name of sample id
    intv : str, optional, default 'interval'
        column name of interval distance, to have overlap calculated
        from the data pass empty string ''
    Returns
    -------
    pandas DataFrame
        DataFrame of overlap data with columns:
        'ovlp_up': str, sample_id of upper sample
        'ovlp_lwr': str, sample_id of lower sample
        'ovlp_dist': numeric, distance of overlap
        'pct_ovlp_up': ovlp_dist over interval distance of upper sample
        'pct_ovlp_lwr': ovlp_dist over interval distance of lower sample
    """
    holes = df[hole_col].unique()  # get the unique holes in the df

    cols = [hole_col, dfrom, dto, samp_id1]

    if intv not in df.columns.tolist():
        intpresent = False
    else:
        cols.append(intv)
        intpresent = True

    temp_d = {
        "ovlp_up": [],
        "ovlp_lwr": [],
        "ovlp_dist": [],
        "pct_ovlp_up": [],
        "pct_ovlp_lw": [],
    }

    for dh in holes:  # for each hole determine if there is overlap
        # temp df of each hole
        temp = df.loc[df[hole_col] == dh, cols]
        # sort by start depth
        temp = temp.sort_values(by=dfrom)
        # reset index
        temp.reset_index(inplace=True)

        # traverse sub-df row by row
        for idx in range(len(temp) - 1):
            # if current row "dto" greater than next row "dfrom" flag it
            if temp.loc[idx, dto] > temp.loc[idx + 1, dfrom]:
                # get interval distance of the two rows
                if intpresent:
                    intv_0 = temp.loc[idx, intv]
                    intv_1 = temp.loc[idx + 1, intv]
                else:
                    intv_0 = temp.loc[idx, dto] - temp.loc[idx, dfrom]
                    intv_1 = temp.loc[idx + 1, dto] - temp.loc[idx + 1, dfrom]

                # calculate overlap distance
                temp_diff = temp.loc[idx, dto] - temp.loc[idx + 1, dfrom]

                # save to dictionary
                temp_d["ovlp_up"].append(temp.loc[idx, samp_id1])
                temp_d["ovlp_lwr"].append(temp.loc[idx + 1, samp_id1])
                temp_d["ovlp_dist"].append(temp_diff)
                temp_d["pct_ovlp_up"].append(temp_diff / intv_0)
                temp_d["pct_ovlp_lw"].append(temp_diff / intv_1)
    overlap_idx = pd.DataFrame(temp_d)
    return overlap_idx


def id_ovlp_to_drop(
    df,
    pct_up="pct_ovlp_up",
    pct_lwr="pct_ovlp_lw",
    samp_up="ovlp_up",
    samp_lwr="ovlp_lwr",
    ovlp_over_cutoff=None,
):
    """Identifies which of overlaping samples to drop
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing upper and lower sample names,
        percent of overlap for both upper and lower samples
    pct_up : str, optional, default 'pct_ovlp_up'
        column name of overlap percent of upper sample
    pct_lwr : str, optional, default 'pct_ovlp_lw'
        column name of overlap percent of lower sample
    samp_up : str, optional, default 'ovlp_up'
        column name of upper sample name
    samp_lwr : str, optional, default 'ovlp_lwr'
        column name of lower sample name
    ovlp_over_cutoff : list_like, optional, default None
        optional, boolean list to mask input df by, if None
        df will be used, if list passed df will be masked
        by ovlp_over_cutoff, keeping only True indexes
    Returns
    -------
    pandas DataFrame
        DataFrame of overlap data with columns
    """
    if ovlp_over_cutoff is not None:
        temp = df[ovlp_over_cutoff].copy().reset_index(drop=True)
    else:
        temp = df.copy().reset_index(drop=True)

    temp["sample_to_drop"] = np.where(
        temp[pct_up] > temp[pct_lwr], temp[samp_up], temp[samp_lwr]
    )

    return temp


def drop_metals_aba_dup(df, which="metals", ignore_na=True):
    """Drops rows based on duplicates in either the Metals or ABA
       subset of columns.
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data, from which duplicates will be dropped
    which : str or list-like, optional, default 'Metals'
        string, 'Metals' or 'ABA' indicates which group to drop
        if list-like, must be a list of start and end column names
    ignore_na : bool, optional, default True
        whether to ignore rows with all nan values when droping duplicates
    custom_range : list-like, optional, default None
        length 2 list of strings for begin and end column names to subset,
        subset is inclusive on both ends
    Returns
    -------
    pandas DataFrame
        returns a copy of the dataframe with duplicates dropped
    """
    # predetermined ranges for metal and aba
    if which.lower() == "metals":
        # slice indexes from columns
        a, b = df.columns.str.lower().slice_locs("metals", "zr_ppm")
        cols = df.iloc[0, a + 1 : b].index
    elif which.lower() == "aba":
        a, b = df.columns.str.lower().slice_locs("aba", "metals")
        cols = df.iloc[0, a + 1 : b - 1].index
    elif isinstance(which, list) and len(which) == 2:
        a, b = df.columns.str.lower().slice_locs(which[0], which[1])
        cols = df.iloc[0, a:b]
    else:
        raise ValueError(
            f"""which must be either 'metals', 'aba', or list of length 2,
            {which} is not valid"""
        )

    if ignore_na:
        return df[(~df.duplicated(subset=cols)) | df[cols].isna().all(1)].reset_index(
            drop=True
        )
    else:
        return df.drop_duplicates(subset=cols).reset_index(drop=True)


def get_discontinuity(exp_df, holeid="drill_hole", dfrom="from", dto="to"):
    """Finds any discontinuities in downhole drill record
    Parameters:
    -----------
    exp_df - pandas dataframe
        contains at minimum holeid, sample start, and sample end
    holeid - str, default 'drill_hole'
        column name of hole id column
    dfrom - str, default 'from'
        column name of sample start column, column needs to be numeric
    dto - str, default 'to'
        column name of sample end column, column needs to be numeric
    Returns:
    --------
    discontinuity - pandas dataframe
        with observations where discontinuities are present. Columns = [holeid,dto,dfrom],
        'to' column is start of discontinuity, 'from' column is end of discontinuity
    """

    # instantiate empty dataframe
    discontinuity = pd.DataFrame()

    # loop through for each holeid in df
    for hole in exp_df[holeid].unique():
        # make temp copy
        temp = exp_df[exp_df[holeid] == hole].copy().reset_index(drop=True)
        # sort df by start column, order matters
        temp.sort_values(by=dfrom, inplace=True)
        # shift dfrom column by -1 to line up with dto column
        temp[dfrom] = temp[dfrom].shift(-1)
        # drop resultant NaN row
        temp.dropna(inplace=True)
        # if realigned dfrom =/= dto mark it
        temp["match"] = temp[dfrom] == temp[dto]
        # record only non-matches
        discontinuity = pd.concat([discontinuity, temp[temp.match == 0]])
    # reindex columns for readability
    return discontinuity.reindex(columns=[holeid, dto, dfrom])


def get_bounds(df):
    """Gets the overall min and overall max of a df
    Parameters
    ----------
    df : pandas DataFrame
        all numeric columns
    Returns
    -------
    tuple
        tuple of min and max values, length 2
    """
    return df.min().min(), df.max().max()


def no_overlapping(x1, x2, y1, y2):
    """Tests for overlap between two sets of numerical values
    Parameters
    ----------
    x1 : numerical
        minimum of group 1
    x2 : numerical
        maximum of group 1
    y1 : numerical
        minimum of group 2
    y2 : numerical
        maximum of group 2
    Returns
    -------
    bool
        True if groups do not overlap, else false
    """
    return max(x1, y1) >= min(x2, y2)  # type: ignore


def find_overlap(minval, maxval, targetfrom, targetto):
    """From two series finds where there is overlap with a min and max value
    Parameters
    ----------
    minval : numerical
        minimum value
    maxval : numerical
        maximum value
    targetfrom : list-like
        numerical values for lower bound
    targetto : list-like
        numerical values for upper bound
    Returns
    -------
    tuple
        two list-like objects with boolean values
    """
    return np.where(targetfrom < maxval, True, False), np.where(
        targetto > minval, True, False
    )


def test_continuity(
    discontinuity,
    env_holes,
    expholeid="drill_hole",
    expfrm="from",
    expto="to",
    envholeid="drill_hole",
    envfrm="from",
    envto="to",
):
    """Given two datasets with sample distance data, one with known discontinuities,
        identifies if and where the second dataset has samples from within those
        discontinuities
    Parameters
    ----------
    discontinuity : pandas DataFrame
        dataframe with known discontinuities, has at minimum hole_id, sample_start,
        sample_end
    env_holes : pandas DataFrame
        test dataframe, discontinuities unknown, has at minimum hole_id, sample_start,
        sample_end
    expholeid : str, optional
        column name of hole_id column for discontinuity df, by default 'drill_hole'
    expfrm : str, optional
        column name of sample_start column for discontinuity df, by default 'from'
    expto : str, optional
        column name of sample_enc column for discontinuity df, by default 'to'
    envholeid : str, optional
        column name of hole_id column for env_holes df, by default 'drill_hole'
    envfrm : str, optional
        column name of sample_start column for env_holes df, by default 'from'
    envto : str, optional
        column name of sample_enc column for env_holes df, by default 'to'
    Returns
    -------
    no_data_env : list
        list of rows from env_holes df that have discontinuities,
        includes columns of sampleid, from, and to
    no_data_exp : list
        list of rows from discontinuity df that have discontinuities,
        includes columns of sampleid, from, and to
    as a tuple with (no_data_env,no_data_exp)
    """
    # instantiate two empty lists
    no_data_env = []
    no_data_exp = []

    # loop through each hole_id in the discontinuity df
    for hole in discontinuity[expholeid].unique():
        # print working hole for tracking
        print(f"Working on {hole}")
        # make copy of subset of each df
        tempexp = discontinuity[discontinuity[expholeid] == hole].copy()
        tempenv = env_holes[env_holes[envholeid] == hole].copy()
        # get maximum and minimum numerical values from the two dfs
        exp_min, exp_max = get_bounds(tempexp[[expfrm, expto]])
        env_min, env_max = get_bounds(tempenv[[envfrm, envto]])
        # determine if any overlap b/t the dfs present
        if no_overlapping(exp_min, exp_max, env_min, env_max):
            # if no overlap skip this iteration
            print("Distances do not overlap\n")
            continue
        # isolate observations in test df where potential overlap could exist
        # (reduces processing time)
        tempenv["under"], tempenv["over"] = find_overlap(
            exp_min, exp_max, tempenv[envfrm], tempenv[envto]
        )
        tempenv["either"] = np.where(
            (tempenv.under == 1) | (tempenv.over == 1), True, False
        )
        # make subset of observations where potential overlap could exist
        tempenv = tempenv[tempenv["either"] == True].copy().reset_index(drop=True)
        # loop through each observation in test df
        for idx, row in tempenv.iterrows():
            # for each observaiton in test df loop through known discontinuities
            for idx_x, row_x in tempexp.iterrows():
                # test if overlap exists on observation by observation level
                if not no_overlapping(
                    row[envfrm], row[envto], row_x[expto], row_x[expfrm]
                ):
                    # if overlap append to list
                    no_data_env.append(row)
                    no_data_exp.append(row_x)
    # return any overlaps that exist
    return no_data_env, no_data_exp


def rename_cols(renamedf, super_sub_scriptreplace):
    """Renames columns in a dataframe to be more pythonic
    Parameters
    ----------
    renamedf : pandas DataFrame
        dataframe to be renamed
    super_sub_scriptreplace : dict
        dictionary of characters to replace in column names,
        default replaces superscripts and subscripts with
        their respective characters, e.g. H₂O becomes H2O
    Returns
    -------
    None
        modifies dataframe in place
    """
    # this function replaces select characters in column names
    # with more pythonic characters inclusing only alphanumeric and _
    # characters and makes all characters lowercase
    for col in renamedf.columns:
        renamedf[col] = renamedf[col].str.replace("%", "pct")
        renamedf[col] = renamedf[col].str.replace("(", "", regex=False)
        renamedf[col] = renamedf[col].str.replace(")", "", regex=False)
        renamedf[col] = renamedf[col].str.replace(r"[^\w\d_]", "_", regex=True)
        renamedf[col] = renamedf[col].str.lower()
        for k, v in super_sub_scriptreplace.items():
            renamedf[col] = renamedf[col].str.replace(k, v, regex=True)
        renamedf[col] = renamedf[col].replace(r"_{2,}", "_", regex=True)
