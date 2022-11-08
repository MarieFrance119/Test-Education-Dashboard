#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: MFLB
"""

# Librairies

# Data Manipulation
import pandas as pd
import numpy as np

# Datavisualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler

# Import KMeans pour clustering
from sklearn.cluster import KMeans



def pre_process(data) :
    """

    Parameters
    ----------
    data : datafram from uploaded file.

    Returns
    -------
    df : dataframe with actionnable indicators and target.

    """
    
    # actionnable indicators
    action = ["Dalc", "Walc", "absences", "studytime"]
    
    # target
    target = ["FinalGrade"]
    
    # Feature engineering
    
    
    # Get dataframe
    df = data[action+target]
    
    return df


def feat_eng(df) :
    """
    Function to get a dataframe with new variables

    Parameters
    ----------
    df : dataframe to do feature engineering, 
     - score_abs : score to scale absences between 1 and 5
     - alc : mean cunsomption alcohol between weekday consumption and weekend
     consumption
     - score_study : inverse of studytime , 1 corresponds to students with 
     higher studytime and 4 students with lower studytime
     - ImprovabilityScore : score to indicate the improvability to help 
     students by using actionnable indicators

    Returns
    -------
    df1 : dataframe.

    """
    df1 = df.copy()
    
    # score_abs creation
    med_abs = df["absences"].median()
    q1_abs = df["absences"].quantile(0.25)
    q3_abs = df["absences"].quantile(0.75)
    fence_high = q3_abs + 1.5 * (q3_abs - q1_abs)
    
    # Conditions list
    conditions_list = [
    (df["absences"] >= fence_high),
    (df["absences"] >= q3_abs),
    (df["absences"] >= med_abs),
    (df["absences"] >= q1_abs),
    (df["absences"] >= df["absences"].min()),
    ]

    # Choice list
    # higher the absences number is, higher is the score
    choicelist = [5, 4, 3, 2, 1]
    df1["score_abs"] = np.select(conditions_list, choicelist)
    
    # Alcohol consumption
    df1["alc"] = (df1["Dalc"] + df1["Walc"]) / 2
    
    # On intervertit les valeurs de studytime
    # en créant une nouvelle variable score_study
    d = {1: 4, 2: 3, 3: 2, 4: 1}
    df1["score_study"] = df1["studytime"].map(d)
    
    # "ImprovabilityScore" feature creation
    df1["ImprovabilityScore"] = df1["alc"] + df1["score_study"] + df1["score_abs"]
    
    return df1


def clustering(df, col_for_cluster) :
    """
    
    Function to cluster data in 5 groups with KMeans
    
    Parameters
    ----------
    df : dataframe with data to clusters.
    col_for_cluster : variables to use for clustering

    Returns
    -------
    df1 : dataframe with number of cluster (= label) for each row.

    """
    
    # Features to use for clustering
    X = df[col_for_cluster]
    
    # Standardisation
    st_scal = StandardScaler().fit(X)
    X_st = st_scal.transform(X)
    
    # initialisation aléatoire et unique
    cls = KMeans(n_clusters=5, random_state=1)
    # n_init = 10 (default)
    # init = ''k-means++' (default)

    # Model fit
    cls.fit(X_st)

    # get labels
    labels = cls.labels_
    
    df1 = df.copy()
    # add labels in dataframe
    df1["label"] = labels
    
    # change label number in order that label 1 is the biggest cluster
    df1["label"].replace(
    df1["label"].value_counts().index, [1, 2, 3, 4, 5], inplace=True)
    
    return df1


def perc_cluster_repartition(df_with_col_label):

    """
    Function to visualize students distribution per cluster
    with percentage (= labels)

    - Arguments :
        - df_with_col_label : dataframe afeter clustering, with label

    - Display :
        - barplot for students distribution per cluster

    """

    # Visualisation sous forme de barplot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x=df_with_col_label["label"].value_counts().index,
        y=df_with_col_label["label"].value_counts().values, 
        palette="tab20",
    )
    plt.title(
        "Students distribution per groups ", 
        size=14)

    # ajout des étiquettes pour afficher valeurs de pourcentages
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100*p.get_height()/
                                      df_with_col_label.shape[0])
        ax.annotate(
           percentage,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=12,
        )
       
    plt.show()

   # return df_with_col_label["label"].value_counts()


def plot_var_cluster(df_with_col_label, col_labels, col):

    """
    Function to visualize features distribution per cluster

    - Arguments :
        - df_with_col_label : dataframe with label feature
        - col_labels : list of variables for which we want
        to display distribution 
        - col : variable or which we want to display distribution

    - Display :
        - 2 figures for each selected feature per cluster
            - barplot of feature mean
            - boxplot 
    """
    
    X = df_with_col_label[col_labels]
    
    # On regroupe par label
    X_labeled = X.groupby("label").mean()
    
    
    # Initialisation de la figure
    f, axes = plt.subplots(1, 2, figsize=(15, 4))

    # boxplot de la variable selon le cluster
    sns.boxplot(
            y="label",
            x=col,
            data=df_with_col_label,
            orient="h",
            ax=axes[0],
            palette="tab20",
        )

    # barplot de la valeur moyenne de la variable selon le cluster
    sns.barplot(
            y=X_labeled.index,
            x=col,
            data=X_labeled,
            orient="h",
            ax=axes[1],
            palette="tab20",
        )

    plt.title("Mean of {} for each group".format(col), size=14)
    plt.show()
