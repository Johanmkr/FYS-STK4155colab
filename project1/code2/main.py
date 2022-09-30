from src.utils import *


from src.objects import designMatrix, targetVector, parameterVector
from src.Regression import linearRegression
from src.Resampling import Bootstrap

import plot as PLOT
PLOT.init('on')






from FrankeAnalysis import *
from terrainAnalysis import *



all_pts = ['B', 'C', 'D', 'E', 'F', 'G']

def runparts(parts, all_pts=all_pts):
    pts = []
    for part in parts:
        pt = part.strip().upper().replace('PT', '')
        if pt == 'ALL':
            pts = all_pts
            break
        else:
            assert pt in all_pts
            pts.append(pt)

    for pt in pts:
        eval(f'pt{pt}()')


try:
    dummy = sys.argv[1]
    parts = sys.argv[1:]
except IndexError:
    parts = input('What parts? ').replace(',', ' ').split()

runparts(parts)