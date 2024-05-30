# fmt: off
# fmt: on --line-length 120

# External modules
import logging
import numpy as np
import pandas as pd
from gurobipy import GRB, Model
import gurobipy as gp
import time
import json

# from config import *
import config as cfg
import functions as fn


# Import prices
dfAll = pd.read_csv(f"{cfg.pathData}pricevectors.csv")
# Set prices to Numpy Array
npAll = dfAll.to_numpy()
npAll = np.delete(npAll, obj=0, axis=1)
npEU = npAll[:][:, 0:24].flatten()
npED = npAll[:][:, 24:48].flatten()
npSpot = npAll[:][:, 48:72].flatten()
npLU = npAll[:][:, 72:96].flatten()
npLD = npAll[:][:, 96:120].flatten()


lst = [npEU, npED, npSpot, npLU, npLD]
cols = ["EU", "ED", "Spot", "LU", "LD"]
npPrices = np.stack(lst)


dfCov = pd.DataFrame(np.cov(npPrices.astype(float), rowvar=True), index=None, columns=cols)
dfCov.insert(loc=0, column="Names", value=cols)
dfCorr = pd.DataFrame(np.corrcoef(npPrices.astype(float), rowvar=True), index=None, columns=cols)
dfCorr.insert(loc=0, column="Names", value=cols)
dfMean = pd.DataFrame(np.mean(npPrices, axis=1), columns=["Mean"])
dfMean.insert(loc=0, column="Names", value=cols)
dfVar = pd.DataFrame(np.var(npPrices, axis=1), columns=["Var"])
dfVar.insert(loc=0, column="Names", value=cols)
dfCov.round(4).to_csv(f"{cfg.pathData}Cov.csv", index=False)
dfCorr.round(4).to_csv(f"{cfg.pathData}Corr.csv", index=False)
dfMean.round(4).to_csv(f"{cfg.pathData}Mean.csv", index=False)
dfVar.round(4).to_csv(f"{cfg.pathData}Var.csv", index=False)


#configList = ['1-1-1', '1-2-4', '1-5-25', '1-10-100', '1-15-225', '1-20-400', '1-25-625','1-27-730']
configList = ['1-1-1 0', '1-2-4 0', '1-2-4 1', '1-2-4 2','1-5-25 0', '1-5-25 1', '1-5-25 2', '1-10-100 0', '1-10-100 1', '1-10-100 2', '1-15-225 0', '1-15-225 1', '1-15-225 2', '1-20-400 0', '1-20-400 1', '1-20-400 2','1-25-625 0', '1-25-625 1', '1-25-625 2', '1-27-730 0', '1-27-730 1', '1-27-730 2', '1-27-730 160', '1-27-730 1364', '1-27-730 2972']

for config in configList:
    dfProbabilities = pd.read_csv(f"{cfg.pathData}{config}/Probabilities.csv")
    # Set probabilities to Numpy Array
    prob = dfProbabilities.to_numpy().ravel()
    # Import prices
    dfPricesEarly = pd.read_csv(f"{cfg.pathData}{config}/PricesEarly.csv")
    dfPricesSpot = pd.read_csv(f"{cfg.pathData}{config}/PricesSpot.csv")
    dfPricesLate = pd.read_csv(f"{cfg.pathData}{config}/PricesLate.csv")
    # Set prices to flat Numpy Arrays for Cov and Corr
    pEUflat = (dfPricesEarly.to_numpy())[:][:, 0:cfg.hourCount].flatten()
    pEDflat = (dfPricesEarly.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount].flatten()
    pSpotflat = dfPricesSpot.to_numpy().flatten()
    pLUflat = (dfPricesLate.to_numpy())[:][:, 0:cfg.hourCount].flatten()
    pLDflat = (dfPricesLate.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount].flatten()
    pListFlat = [pEUflat, pEDflat, pSpotflat, pLUflat, pLDflat]
    pPricesFlat = np.stack(pListFlat)
    # Non-flat Numpy Arrays for Mean
    pEU = (dfPricesEarly.to_numpy())[:][:, 0:cfg.hourCount]
    pED = (dfPricesEarly.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount]
    pSpot = dfPricesSpot.to_numpy()
    pLU = (dfPricesLate.to_numpy())[:][:, 0:cfg.hourCount]
    pLD = (dfPricesLate.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount]
    pList = [pEU, pED, pSpot, pLU, pLD]

    dfCov = pd.DataFrame(np.cov(pPricesFlat.astype(float), rowvar=True), index=None, columns=cols)
    dfCov.insert(loc=0, column="Names", value=cols)
    dfCorr = pd.DataFrame(np.corrcoef(pPricesFlat.astype(float), rowvar=True), index=None, columns=cols)
    dfCorr.insert(loc=0, column="Names", value=cols)
    dfMean = pd.DataFrame([np.average(np.average(price, axis=0, weights=prob)) for price in pList], columns=[config])
    dfMean.insert(loc=0, column="Names", value=cols)
    dfVar = pd.DataFrame(np.var(pPricesFlat, axis=1), columns=[config])
    dfVar.insert(loc=0, column="Names", value=cols)
    dfCov.round(4).to_csv(f"{cfg.pathData}{config}/Cov.csv", index=False)
    dfCorr.round(4).to_csv(f"{cfg.pathData}{config}/Corr.csv", index=False)
    dfMean.round(4).to_csv(f"{cfg.pathData}{config}/Mean.csv", index=False)
    dfVar.round(4).to_csv(f"{cfg.pathData}{config}/Var.csv", index=False)

dfMeans = pd.read_csv(f"{cfg.pathData}Mean.csv")
dfVars = pd.read_csv(f"{cfg.pathData}Var.csv")
dfVars2 = pd.read_csv(f"{cfg.pathData}Var.csv")

for config in configList:
    dfMean = pd.read_csv(f"{cfg.pathData}{config}/Mean.csv")
    dfMeans = dfMeans.merge(dfMean, on="Names")

    dfVar = pd.read_csv(f"{cfg.pathData}{config}/Var.csv")
    dfVars = dfVars.merge(dfVar, on="Names")

dfMeans.round(4).to_csv(f"{cfg.pathData}Means.csv", index=False)
dfVars.round(4).to_csv(f"{cfg.pathData}Vars.csv", index=False)
dfVars2.round(4).to_csv(f"{cfg.pathData}Vars2.csv", index=False)



if fn.fileExists(cfg.pathJSON):
    with open(f"{cfg.pathJSON}") as f:
        data = json.load(f)


dfResults = pd.read_json(f"{cfg.pathJSON}", orient='columns')
dfResults.round(2).to_csv(f"{cfg.pathData}results.csv")
