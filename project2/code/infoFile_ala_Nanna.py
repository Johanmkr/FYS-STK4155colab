import os
import pandas as pd
from collections import OrderedDict


'''
Create an -md.file with a table decribing e.g. .pdf-files in a figure folder.
'''


def init(md_filename="README", pkl_filename="info", path="/../output/figures/"):
    global MD_FILE, PKL_FILE, PATH_str
    MD_FILE = md_filename.replace(".md", "") + ".md"
    PKL_FILE = pkl_filename.replace(".pkl", "") + ".pkl"
    PATH_str = path


HERE = os.path.abspath(".")
try: 
    PATH = HERE + PATH_str
except NameError:
    init()
    PATH = HERE + PATH_str



deleteme = []
addme = []


def sayhello():

    try:
        infopd = pd.read_pickle(PATH + PKL_FILE)
    except FileNotFoundError:
        dummy = {"dummy.jpg": {"note": "delete me"}}
        dummypd = pd.DataFrame.from_dict(dummy, orient='index')
        dummypd.to_pickle(PATH + PKL_FILE)
        infopd = pd.read_pickle(PATH + PKL_FILE)

    global INFO
    INFO = infopd.transpose().to_dict()


def set_file_info(filename, note=None):
    INFO[filename] = {'note':note}


def omit_category():
    infopd = pd.DataFrame.from_dict(INFO, orient='index')   



def omit_file(filename):
    deleteme.append(filename)


def additional_information(bulletpoint):
    addme.append(bulletpoint)


def update(additional_information=addme, header=f"Description of plots in {PATH_str}"):
    info = OrderedDict(sorted(INFO.items(), key=lambda i:i[0].lower()))         # sort alphabetically
    infopd = pd.DataFrame.from_dict(info, orient='index')                       # create dataframe
    for filename in deleteme:                                                   # delete files in 'deleteme'-list
        infoT = infopd.transpose()
        try:
            infoT.pop(filename)
        except KeyError:
            print(f"There is no saved information about {filename} - please remove delete-command.")
        infopd = infoT.transpose()
    infopd.to_pickle(PATH + PKL_FILE)                                           # save in .pkl
    infopd.to_markdown(PATH + MD_FILE)                                          # create nice table in .md


    with open(PATH + MD_FILE, 'a') as file:
        file.write('\n\n\n')
        file.write(f'# {header}')
        file.write('\n\n')

        if len(additional_information)>0:
            file.write('\n## Additional information:\n\n')
            for line in additional_information:
                file.write(f'* {line}\n')

    print(f'\nSuccessfully written information to \n    {PATH_str}{MD_FILE}.\n')




