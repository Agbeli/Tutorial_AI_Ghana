from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 




if __name__ == "__main__":
    ###################################### generate 2d classification dataset #########################################################
    X, y = datasets.make_moons(n_samples=100, noise=0.1)
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    df["names"] = np.where(df["label"]== 1,"positive","negative")
    colors = {"positive":'green', "negative":'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('names')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.title("labeled dataset",fontsize="medium")
    plt.savefig("labeled.png")
    plt.show()

    ###################################### Unlabeled datasets #######################################################################
    X ,y = datasets.make_regression(n_samples=100,n_features=2)
    plt.scatter(X[:,0],X[:,1],marker="x")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("unlabeled dataset")
    plt.savefig("unlabeled.png")
    plt.show()