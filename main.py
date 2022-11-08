#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MFLB

"""

# Librairies

# Data manipulation
import pandas as pd

# Streamlit deployment
import streamlit as st

# python file with functions
import functions as f

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns


#magic uses st.write
# Title
st.write('# Education dashboard V1')
# Explaination
st.markdown(''' 
            This is a dashboard showing 
            - the *ImprovabilityScore* and the *FinalGrade* of students, categorized in 5 groups
            - the Repartition of the students in the 5 groups
            - the Characteristics of these 5 groups''')

# avoid displaying warning message
st.set_option('deprecation.showPyplotGlobalUse', False)

# File uploader
file = st.file_uploader("Load your data")

# Initialization of placeholders
file_placeholder = st.empty()
success = st.empty()
submit_placeholder = st.empty()
submit=False

# When uploading file
if file is not None :
    with st.spinner("Data loading..."):  
        path = file
        #file_placeholder.image(img)

    # Submit button creation
    submit = submit_placeholder.button("Launch data analyse")

# When clicking on submit button
if submit :
    with st.spinner('Searching...'):    
        submit_placeholder.empty()
        data = pd.read_csv(path)

        # List of variables to use 
        col_for_cluster = ["score_study", "score_abs", "alc", "FinalGrade"]

        # Add label variable
        col_labels = col_for_cluster + ["label"]
        
        # Display Improvability and FinalGrade header
        st.header('ImprovabilityScore and FinalGrade')
        
        # Initialize the figure
        fig, ax = plt.subplots()
        
        # Scatterplot with number of clusters and labels
        sns.scatterplot(
            x="FinalGrade",
            y="ImprovabilityScore",
            data=f.clustering(
                f.feat_eng(
                    f.pre_process(data)
                    ), 
                col_for_cluster
                ),
            hue="label",
            palette="tab20",
            )
        plt.xlabel("FinalGrade", size=14)
        plt.ylabel("ImprovabilityScore", size=14)
        # Plot the figure on streamlit interface
        st.pyplot(fig)
        
        
        # Groups repartition
        st.header('Groups repartition')
        
        # Get the figure
        fig2 = f.perc_cluster_repartition(
                    f.clustering(
                    f.feat_eng(
                        f.pre_process(data)
                        ), 
                    col_for_cluster
                    )
                    )
        # Plot the figure on streamlit interface
        st.pyplot(fig2)

        # Groups characteristics
        st.header('Groups characteristics')
        
        # Display boxplot and barplot for each variable
        for col in col_for_cluster :
            fig = f.plot_var_cluster(
                            f.clustering(
                                f.feat_eng(
                                    f.pre_process(data)
                                    ), 
                                col_for_cluster
                                ), 
                            col_labels,col
                            )
            # Plot the figure on streamlit interface
            st.pyplot(fig)
