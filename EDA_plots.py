# collection of scripts to preprocess raw data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, calendar
from scipy import stats



# fill area under the line -------------------------------------------------------------------------
def fill_under_lines(ax=None, alpha=.2, **kwargs):
    if ax is None:
        ax = plt.gca()
    for line in ax.lines:
        x, y = line.get_xydata().T
        ax.fill_between(x, 0, y, color=line.get_color(), alpha=alpha, **kwargs)


# projects launched --------------------------------------------------------------------------------
def timeseries(df):
    # number of projects
    ts = df.set_index(pd.DatetimeIndex(df['launched_at'])).sort_index().usd_pledged.resample('MS').count()
    plt.figure(figsize=(10,3))
    sns.lineplot(ts)
    plt.xlabel('Launch date', fontsize=12)
    plt.ylabel('Number of projects', fontsize=12)
    plt.title('Number of projects launched on Kickstarter, 2009-2023', fontsize=16)
    fill_under_lines(alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plotting the cumulative amount pledged on Kickstarter
    ts = df.set_index(pd.DatetimeIndex(df['launched_at'])).sort_index().usd_pledged.resample('MS').sum()
    plt.figure(figsize=(10,3))
    sns.lineplot(ts)
    plt.xlabel('Launch date', fontsize=12)
    plt.ylabel('Amount pledged ($)', fontsize=12)
    plt.title('Pledges on Kickstarter, 2009-2019', fontsize=16)
    fill_under_lines(alpha=0.5)
    plt.tight_layout()
    plt.show()


# barplots -----------------------------------------------------------------------------------------
def barplots(df):
    err = None
    order = ['successful', 'failed']

    # Plotting the average amount pledged to successful and unsuccesful projects
    _, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10,10))

    sns.countplot(data=df, x='state', hue='state', order=order, ax=ax1)
    ax1.set_title('Number of projects')
    ax1.set_xlabel('')

    sns.barplot(data=df, y='name_length', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax2, legend=False)
    ax2.set_title('Median name length')
    ax2.set_xlabel('')
    ax2.set_ylabel('# words') 

    sns.barplot(data=df, y='blurb_length', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax3, legend=False)
    ax3.set_title('Median blurb length')
    ax3.set_xlabel('')
    ax3.set_ylabel('# words')

    sns.barplot(data=df, y='usd_goal', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax4, legend=False)
    ax4.set_title('Median project pledge goal')
    ax4.set_xlabel('')
    ax4.set_ylabel('pledge goal [$]')

    sns.barplot(data=df, y='campaign_days', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax5, legend=False)
    ax5.set_title('Median campaign length')
    ax5.set_xlabel('')
    ax5.set_ylabel('days')

    sns.barplot(data=df, y='preparation_days', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax6, legend=False)
    ax6.set_title('Median preparation time')
    ax6.set_xlabel('')
    ax6.set_ylabel('days')

    sns.barplot(data=df, y='usd_pledged', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax7, legend=False)
    ax7.set_title('Median pledges per project')
    ax7.set_xlabel('')
    ax7.set_ylabel('pledged [$]')

    sns.barplot(data=df, y='backers_count', x='state', hue='state',
                estimator='median', errorbar=err, order=order, ax=ax8, legend=False)
    ax8.set_title('Median backers per project')
    ax8.set_xlabel('')
    ax8.set_ylabel('count')

    # Creating a dataframe grouped by staff_pick with columns for failed and successful
    # Normalizes counts by column, and selects the 'True' category (iloc[1])
    counts = df.groupby(['staff_pick', 'state'])['deadline_at'].count().rename('freq')
    freq = counts.div(counts.groupby('state').sum()).round(2).reset_index()
    freq = freq[freq['staff_pick']]

    sns.barplot(freq, y='freq', x='state', hue='state', hue_order=order, order=order, ax=ax9)
    ax9.set_title('Proportion that were "staff picks"')
    ax9.set_xlabel('')
    ax9.set_ylabel('proportion')

    plt.tight_layout()
    plt.show()


# histplots ----------------------------------------------------------------------------------------
def histplots(df, hue=None, hue_order=None, cols=3, kde=False, bins='auto', binwidth=None,
              xlog=None, ylog=None):
    # handle features to plot
    features = df.columns
    if hue is not None:
        features = features.drop(hue)
    
    # handle logarithmic axis
    log = [[False, False] for i in range(len(features))]
    if xlog is not None:
        log = [[i, log[i][1]] for i in xlog]
    if ylog is not None:
        log = [[log[i][0], i] for i in ylog]

    # create subplot
    ncol = cols
    nrow = math.floor((len(features) + ncol - 1) / ncol)
    _, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8*ncol, 4*nrow))

    # go over a list of features
    for i in range(len(features)):
        # compute an appropriate index (1d or 2d)
        ix = np.unravel_index(i, axarr.shape)

        # histplot
        g = sns.histplot(data=df, x=features[i], hue=hue, hue_order=hue_order,
                         bins=bins, binwidth=binwidth, log_scale=log[i], kde=kde,
                         ax=axarr[ix])
        axarr[ix].set_title(features[i])
        axarr[ix].set_ylabel('')
        axarr[ix].set_xlabel('')
    
    # turn off unused axes
    for j in range(i+1,ncol*nrow):
        jx = np.unravel_index(j, axarr.shape)
        axarr[jx].set_axis_off()

    plt.tight_layout()
    plt.show()


# box by year --------------------------------------------------------------------------------------
def bar_by_feature(df, feature, doSort=None, palette=None):
    # Creating a dataframe grouped by category with columns for failed and successful (FeatureSuccessRate)
    feat_df = pd.get_dummies(df.set_index(feature).state).groupby(feature).sum()

    # Plotting
    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10,12))

    # Project Success Rate
    psr = feat_df.div(feat_df.sum(axis=1), axis=0).successful
    if doSort is not None:
        if doSort == 'SuccessRate':
            psr = psr.sort_values(ascending=False)
        else:
            psr = psr.reindex(doSort)
    psr_order = psr.index

    sns.barplot(x=psr, y=psr.index, hue=psr.index, ax=ax1, palette=palette)
    ax1.set_title('Proportion of successful projects')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    _ = df.groupby(feature)[feature].count()[psr_order]
    sns.barplot(x=_, y=_.index, hue=_.index, ax=ax2, palette=palette)
    ax2.set_title('Number of projects')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    _ = df.groupby(feature).usd_goal.median()[psr_order]
    sns.barplot(x=_, y=_.index, hue=_.index, ax=ax3, palette=palette)
    ax3.set_title('Median project goal ($)')
    ax3.set_xlabel('')
    ax3.set_ylabel('')

    _ = df.groupby(feature).usd_pledged.median()[psr_order]
    sns.barplot(x=_, y=_.index, hue=_.index, ax=ax4, palette=palette)
    ax4.set_title('Median pledged per project ($)')
    ax4.set_xlabel('')
    ax4.set_ylabel('')

    _ = df.groupby(feature).backers_count.median()[psr_order]
    sns.barplot(x=_, y=_.index, hue=_.index, ax=ax5, palette=palette)
    ax5.set_title('Median backers per project')
    ax5.set_xlabel('')
    ax5.set_ylabel('')

    _ = df.groupby(feature).pledge_per_backer.median()[psr_order]
    sns.barplot(x=_, y=_.index, hue=_.index, ax=ax6, palette=palette)
    ax6.set_title('Median pledged per backer ($)')
    ax6.set_xlabel('')
    ax6.set_ylabel('')
    
    plt.suptitle(feature, fontsize='xx-large')
    plt.tight_layout()
    plt.show()


# correlation matrix -------------------------------------------------------------------------------
def corr_matrix(df, features, thresh: float=None, col: str=None):
    """    Display a heatmap from a given dataset

    Args:
        data (dataset): dataframe containing columns to be correlated with each other
        thresh (float): threshold correlation value defines which r's should be annotated
    
    Returns:
        g (graph)
    """
    # Generate a custom diverging colormap
    cmap = sns.color_palette("vlag", as_cmap=True)

    # handle annot parameter
    annot = False if thresh is None else True

    # handle possible col parameter
    if col is None:
        n_cols = 1
        df = df[features]
    else:
        df = df[features + [col]]
        n_cols = df[col].nunique()
        grp_by = df[col].unique()

    # loop columns
    plt.figure(figsize=(7*n_cols, 7))
    for i in range(n_cols):
        # filter df according to grp_by
        if col is not None:
            df_col = df.loc[df[col] == grp_by[i], df.columns != col]
        else:
            df_col = df

        # Create a correlation matrix & a mask for the lower triangle
        corr = df_col.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = None

        # Draw the heatmap with the mask and correct aspect ratio
        ax = plt.subplot(1, n_cols, i+1)
        sns.heatmap(corr, cmap=cmap, mask=mask, square=True, annot=annot, ax=ax, center=0,
                    xticklabels=features[:-1], yticklabels=['']+features[1:],
                    cbar_kws={"shrink": 0.5, 'label':'Pearson r', 'anchor': (0, 0.45)})
        ax.tick_params(left=False, bottom=False)
        ax.grid(False)
        if col is not None:
            ax.set_title(grp_by[i], fontsize=20)
        
        # add r values if above threshold
        if thresh is not None:
            for t in ax.texts:
                if abs(float(t.get_text()))>=thresh:
                    t.set_text(t.get_text()) #if the value is greater than _thresh_ then I set the text 
                else:
                    t.set_text("") # if not it sets an empty text

    # plot
    plt.tight_layout()
    plt.show()


# scatter/regplot ----------------------------------------------------------------------------------
def pairs(df, features=None, hue=None, kind='reg'):
    hue_order = None
    if hue is None:
        df = df[features]
    else:
        df = df[features + [hue]]
        if hue == 'state':
            hue_order = ['successful', 'failed']

    pp = sns.pairplot(data=df, hue=hue, hue_order=hue_order, height=3, kind=kind,
                 corner=True, plot_kws=dict(scatter_kws=dict(s=2)), )

    plt.tight_layout()
    plt.show()
    

# main -> run EDA wrt distribution -----------------------------------------------------------------
def run(df, *args):
    timeseries(df)
    barplots(df)

    histplots(df[['name_length', 'blurb_length', 'preparation_days', 'campaign_days', 'usd_goal', 'state']],
              hue='state', bins='doane', cols=2, xlog=[False, False, True, False, True])

    bar_by_feature(df, 'category', doSort='SuccessRate', palette='flare')
    bar_by_feature(df, 'country', palette='flare')
    bar_by_feature(df, 'deadline_day', doSort = list(calendar.day_name)[0:], palette='flare')
    bar_by_feature(df, 'deadline_month', doSort=list(calendar.month_name)[1:], palette='flare')

    corr_feat = ['backers_count', 'usd_pledged', 'name_length', 'blurb_length', 'pledge_per_backer']
    corr_matrix(df, corr_feat, thresh=.1)
    corr_matrix(df, corr_feat, col='state', thresh=.1)

    # corr_feat = ['usd_goal', 'name_length', 'blurb_length', 'preparation_days']
    # pairs(df.loc[:1000, :], features=corr_feat, hue='state')


if __name__ == "__main__":
    curdir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(curdir, 'data', "df_preprocessed.csv"))
    run(df)