import os
import pandas as pd
from collections import OrderedDict


'''
Create an -md.file with a table decribing e.g. .pdf-files in a figure folder.
'''


def init(md_filename:str="README", pkl_filename:str="info", path:str="/../output/figures/"):
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

    global INFO, CATEGORIES
    INFO = infopd.transpose().to_dict()

    CATEGORIES = {}
    for fig in INFO.keys():
        for category in INFO[fig].keys():
            # FIXME
            CATEGORIES[category] = category
        break


def define_categories(categories:dict, include_note:bool=True):
    # Need to be called everytime, fix this?
    global CATEGORIES
    CATEGORIES = categories
    if not 'note' in CATEGORIES and include_note:
        CATEGORIES['note'] = 'note' #comment?





def set_file_info(filename:str, **params):
    INFO[filename] = {}
    for category in CATEGORIES.keys():
        try:
            s = str(params[category])
        except KeyError:
            s = None
        INFO[filename][CATEGORIES[category]] = s




def omit_category(category):
    # depricated (is this a word)
    infopd = pd.DataFrame.from_dict(INFO, orient='index')   
    infopd.pop(category)


def omit_file(filename:str):
    deleteme.append(filename)


def additional_information(bulletpoint:str):
    addme.append(bulletpoint)


def update(additional_information:list=addme, header:str=f"Description of plots in {PATH_str}"):
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




