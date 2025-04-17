import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_featur_heatmap(feature_df, save_path, save_name):
    # df_train = pd.DataFrame(data=feature_new, columns=feature_name_new)
    # df_test = pd.DataFrame(data=feature_test_new, columns=feature_name_new)
    # sns.heatmap(df_train.corr(), annot=True, cmap='viridis')
    # sns.heatmap(df_test.corr(), annot=True, cmap='viridis')

    title = save_name[:-4]
    g = sns.clustermap(feature_df.corr(), xticklabels=False, yticklabels=False, cmap='seismic', figsize=(8, 8))
    # g.ax_heatmap.set_title(title)
    # g.fig.subplots_adjust(top=.9)
    g.fig.suptitle(title, y=1)
    # g.fig.subplots_adjust(top=.9)
    plt.savefig(os.path.join(save_path, save_name))
