import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import aic, bic
from sklearn.metrics import r2_score
import patsy
import matplotlib.pyplot as plt

def find_num_components_explaining_90variance(data):
    """
    This function performs a PCA and finds the number of components that
    explain 90 %, 95 % and 99 % of the variance in the data.
    """
    fpca = PCA()
    fpca.fit(data)

    # Get the cumulative explained variance (in %)
    var_cumu = np.cumsum(fpca.explained_variance_ratio_) * 100

    # Percent-variance thresholds we care about
    percent_var = [80, 85, 90, 95, 99]

    for var in percent_var:
        k_all = np.argmax(var_cumu > var)
        print(f"No. of components explaining {var} % of the variance in data: {k_all}")
    #end function find_num_components_explaining_90variance

def reduce_data(ratings_data, num_components,
                components_weights_filename=None):
    """
    This method reduces the data into the specified number of principal components.
    The feelings data is grouped by video name.
    """

    # fitting the data into n components
    col_names = []  # this is to create the column names
    for i in range(num_components):
        col_names.append(f'PC{i+1}')

    fpca = PCA(n_components=num_components)
    vife_red = fpca.fit_transform(ratings_data)

    # Display components relations with features:
    components_weight = pd.DataFrame(
        fpca.components_,
        columns=ratings_data.columns,
        index=col_names
    )
    #display(components_weight)
    components_weight.index.name = None

    if components_weights_filename is not None:
        components_weight.to_excel(
            components_weights_filename,
            # index=False  # Uncomment if you don't want the index in the Excel file
        )

    # Rename the column names as PC1, PC2, ...
    # for i in range(num_components):
    #     vife_red.rename(columns={i: col_names[i]}, inplace=True)

    # Show total explained variance
    total_var = fpca.explained_variance_ratio_.sum() * 100
    print(f'Proportion of variance explained by each of the {num_components} components is: {fpca.explained_variance_ratio_}')
    print(f'Total explained variance by {num_components} components is: {total_var}')

    return vife_red

def components_weight(ratings_data, num_components):
    """
    This method reduces the data into the specified number of principal components.
    The feelings data is grouped by video name.
    """

    # fitting the data into n components
    col_names = []  # this is to create the column names
    for i in range(num_components):
        col_names.append(f'PC{i+1}')

    fpca = PCA(n_components=num_components)
    vife_red = fpca.fit_transform(ratings_data)

    # Display components relations with features:
    components_weight = pd.DataFrame(
        fpca.components_,
        columns=ratings_data.columns,
        index=col_names
    )
    #display(components_weight)
    components_weight.index.name = None

    return components_weight

def display_pc3d(df, save_fig=False):
    """
    Displays a 3D scatter plot of PC1, PC2, PC3, colored by 'scene'.
    
    Parameters:
    - df: dataframe with columns 'PC1', 'PC2', 'PC3', and 'scene'
    - save_fig: if True, saves figure as 'pc3d_plot.png'
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors for each scene
    unique_scenes = df['scene'].unique()
    scene_colors = {
        scene: color for scene, color in zip(unique_scenes, ['royalblue', 'lightsalmon', 'lightskyblue', 'gold', 'red', 'green'])
    }

    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 15), layout='constrained')
    ax = fig.add_subplot(projection='3d')

    # Plot each scene
    for scene, color in scene_colors.items():
        df_scene = df[df['scene'] == scene]
        ax.scatter(df_scene['PC1'], df_scene['PC2'], df_scene['PC3'],
                   c=color, label=scene, s=200)

    # Axis styling
    fonts = {'size': 22}
    ax.view_init(60, -140)
    ax.set_xlabel('PC1', fontdict=fonts)
    ax.set_ylabel('PC2', fontdict=fonts)
    ax.set_zlabel('PC3', rotation=90, fontdict=fonts)
    ax.set_title('The Weights of Each Component Per Scene (90-Second)', fontsize=30)
    ax.set_zlim(-3.5, 3.5)
    ax.xaxis.labelpad = 20.0
    ax.yaxis.labelpad = 20.0
    ax.zaxis.labelpad = 10.0

    # Ticks
    ax.set_xticks(np.arange(-10, 10, 5))
    ax.set_yticks(np.arange(-4, 4, 2))
    ax.set_zticks(np.arange(-3.5, 3.5, 2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    # Add legend
    ax.legend(loc='best', prop={'size': 22})

    # Save figure if needed
    if save_fig:
        fig.savefig('pc3d_plot.png', dpi=300)

    # Show plot
    plt.show()
    plt.close(fig)

from tkinter import font

def display_pc_weights_1(components_weight, is_horizontal=True, save_fig=False):
    """
    Displays the weights (loadings) of all ratings on each principal component.
    
    Parameters:
    - components_weight: DataFrame of shape (ratings x PCs) or (PCs x ratings)
    - is_horizontal: if True, shows PCs on y-axis; if False, PCs on x-axis
    - save_fig: if True, saves the figure as 'pc_weights_plot.png'
    
    Returns:
    - fig, axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sb

    # Set Calibri font
    plt.rcParams['font.family'] = 'Calibri'

    if is_horizontal:
        # Horizontal: ratings on x, PCs on y
        #display(components_weight)

        fig, axes = plt.subplots(figsize=(10, 4))
        sb.heatmap(components_weight, cmap='RdBu_r', ax=axes,
                   yticklabels=['PC1', 'PC2', 'PC3'],
                   annot=True, annot_kws={"fontsize": 30}, vmin=-1, vmax=1)

        axes.set_xlabel('Features', fontsize=50)
        axes.set_ylabel('Principal Component', fontsize=50)

    else:
        # Vertical: transpose → ratings on y, PCs on x
        components_weight = components_weight.transpose()
        #display(components_weight)

        fig, axes = plt.subplots(figsize=(5, 8))
        sb.heatmap(components_weight, cmap='RdBu_r', ax=axes,
                   xticklabels=['PC1', 'PC2', 'PC3'],
                   annot=True,  fmt=".2f", annot_kws={"size": 20}, vmin=-1, vmax=1)

        colorbar = axes.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=15) 

        axes.set_ylabel('Features', fontsize=14)
        axes.set_xlabel('Principal Component', fontsize=14)
        

    # Ticks style
    axes.tick_params(axis='x', labelrotation=0, labelsize=12)
    axes.tick_params(axis='y', labelrotation=0, labelsize=18)

    # Save figure if needed
    if save_fig:
        fig.savefig('pc_weights_plot.png', dpi=300)

    # Return fig and axes so user can modify after
    return fig, axes
