from src.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as R
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd


# The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# folder paths
here = os.path.abspath(".")
data_path = here + "/../output/data/"
plot_path = here + "/../output/figures/"
latex_path = here + "/../latex/"


CMAP_ACCURACY = "gnuplot2" # "spring"
CMAP_MSE = CMAP_ACCURACY + "_r"




def save_push(fig, pdf_name, save=True, push=True, show=False, tight=False):
    if tight:
        fig.tight_layout()
    file = plot_path + pdf_name.replace('.pdf', '').strip() + ".pdf"
    if save and pdf_name != 'none':
        print(f'Saving plot: {file}')
        fig.savefig(file, bbox_inches="tight")
    if push and pdf_name != 'none':
        os.system(f"git add {file}")
        os.system("git commit -m 'upload plot'")
        os.system("git push")
    if show:
        plt.show()
    else:
        plt.clf()

    plt.close()

def get_last_colour():
    return plt.gca().lines[-1].get_color()



# save useful information
import src.CollectAndSaveInfo as Nanna
Nanna.init()
Nanna.sayhello()
Nanna.define_categories({"method":"method"})  


def set_pdf_info(pdfname:str, **params):
    Nanna.set_file_info(pdfname.replace('.pdf', '') + '.pdf', **params)


def copy_info(info:dict, filename:str):
    params = {} 
    for param, title in Nanna.CATEGORIES.items():
        try:
            params[param] = info[filename][title]
        except KeyError:
            continue
    return params


def update():
    Nanna.update()