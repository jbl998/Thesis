# fmt: off
# fmt: on --line-length 120

# External modules
import logging
import time
import json
import datetime as dt
import numpy as np
import pandas as pd
import requests
from sklearn_extra.cluster import (
    KMedoids,
)  # https://scikit-learn-extra.readthedocs.io/en/stable/_modules/sklearn_extra/cluster/_k_medoids.html#KMedoids


# from config import *
import config as cfg
import functions as fn

# Setup logging and start a logging instance
cfg.setup_logging()
logger = logging.getLogger(__name__)

start = dt.datetime.now()
stg1clusters = 1
stg2clusters = 27  # Number of nodes in stage 2
stg3clusters = 730 #stg2clusters**2  # Number of nodes in stage 3

configuration = f"{stg1clusters}-{stg2clusters}-{stg3clusters}"
logger.info(f"Running configuration '{configuration}'")


# DATA IMPORT
def getPrices():
    logger.info(f"Reading data from EnergiDataService")
    responseFCR = requests.get(url=cfg.fcrURL)
    responseSpot = requests.get(url=cfg.spotURL)
    result = {}
    result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    result["status_code"] = responseFCR.status_code
    if responseFCR.status_code == 200:
        result["dataFCR"] = responseFCR.json()["records"]
        dfDataFCR = pd.DataFrame.from_records(result["dataFCR"])
    if responseSpot.status_code == 200:
        result["dataSpot"] = responseSpot.json()["records"]
        dfDataSpot = pd.DataFrame.from_records(result["dataSpot"])

    # DATA CONVERSION
    logger.info(f"Converting imported data")
    # Convert 'HourDK' to DateTime format
    dfDataFCR['HourDK'] = pd.to_datetime(dfDataFCR['HourDK'], format='ISO8601')

    # Group by 'HourDK', 'PriceArea', 'ProductName', and 'AuctionType', then pivot the table
    grouped = dfDataFCR.groupby(['HourDK', 'PriceArea', 'ProductName', 'AuctionType'])['PriceTotalEUR'].sum().reset_index()
    pivot_table = grouped.pivot_table(index=['HourDK', 'PriceArea'], columns=['ProductName', 'AuctionType'],
                                    values='PriceTotalEUR').reset_index()

    # Flatten the multi-index columns
    pivot_table.columns = pivot_table.columns.map(lambda x: f'{x[0]}, {x[1]}' if x[1] else x[0])

    # Create a range of dates with hourly frequency
    date_range = pd.date_range(start=cfg.start2, periods=len(pivot_table), freq='h')

    pivot_table['Timestamp'] = pd.DataFrame({'Timestamp': date_range})
    pivot_table['Date'] = pivot_table['Timestamp'].dt.date
    pivot_table['Hour'] = pivot_table['Timestamp'].dt.hour
    dfEU = pivot_table.pivot_table(index=['Date'], columns=['Hour'], values='FCR-D upp, D-1 early', fill_value=0,
                                aggfunc='sum', dropna=False).reset_index()
    dfLU = pivot_table.pivot_table(index=['Date'], columns=['Hour'], values='FCR-D upp, D-1 late', fill_value=0,
                                aggfunc='sum', dropna=False).reset_index()
    dfED = pivot_table.pivot_table(index=['Date'], columns=['Hour'], values='FCR-D ned, D-1 early', fill_value=0,
                                aggfunc='sum', dropna=False).reset_index()
    dfLD = pivot_table.pivot_table(index=['Date'], columns=['Hour'], values='FCR-D ned, D-1 late', fill_value=0,
                                aggfunc='sum', dropna=False).reset_index()

    # Create a range of dates with hourly frequency
    date_range = pd.date_range(start=cfg.start2, periods=len(dfDataSpot), freq='h')
    dfDataSpot['Timestamp'] = pd.DataFrame({'Timestamp': date_range})
    dfDataSpot['Date'] = dfDataSpot['Timestamp'].dt.date
    dfDataSpot['Hour'] = dfDataSpot['Timestamp'].dt.hour
    dfSpot = dfDataSpot.pivot_table(index="Date", columns=['Hour'], values='SpotPriceEUR', fill_value=0, aggfunc='sum',
                                    dropna=False).reset_index()

    # Combine DataFrames to 120-axis vectors for 
    dfEarly = dfEU.merge(dfED, on='Date', suffixes=['_EU', '_ED'])
    dfLate = dfLU.merge(dfLD, on='Date', suffixes=['_LU', '_LD'])
    dfAll = dfEarly.merge(dfSpot, on='Date').merge(dfLate, on='Date')
    dfAll.to_csv(f"{cfg.pathData}pricevectors.csv", index=False)
    return dfAll

if fn.fileExists(f"{cfg.pathData}pricevectors.csv"):
    dfAll = pd.read_csv(f"{cfg.pathData}pricevectors.csv")
else:
    dfAll = getPrices()

npAll = dfAll.to_numpy()
npAll = np.delete(npAll, obj=0, axis=1)

def getDistance():
    # Calculate distances between 120 dimension price vectors
    logger.info(f"Calculating distance matrix")
    dim = len(dfAll)
    npDist = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            v1 = dfAll.loc[i, '0_EU':'23_LD'].to_list()
            v2 = dfAll.loc[j, '0_EU':'23_LD'].to_list()
            if i == j:
                npDist[i, j] = 0
            elif i < j:
                npDist[i, j] = np.sqrt(sum((float(v1[k]) - float(v2[k])) ** 2 for k in range(len(v1))))
            else:
                npDist[i, j] = npDist[j, i]

    dates = dfAll['Date'].to_list()
    dfDist = pd.DataFrame(npDist, columns=dates)
    dfDist.insert(0, 'Date', dates)
    dfDist.to_csv(f"{cfg.pathData}distances.csv", index=False)
    return dfDist, npDist

def getDistances():
    # Calculate distance matrices for each stage separately
    logger.info(f"Calculating distance matrix")
    dim = len(dfAll)
    npDist_stg1 = np.zeros((dim, dim))
    npDist_stg2 = np.zeros((dim, dim))
    npDist_stg3 = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            stg1_v1 = dfAll.loc[i, '0_EU':'23_ED'].to_list()
            stg1_v2 = dfAll.loc[j, '0_EU':'23_ED'].to_list()
            stg2_v1 = dfAll.loc[i, '0':'23'].to_list()
            stg2_v2 = dfAll.loc[j, '0':'23'].to_list()
            stg3_v1 = dfAll.loc[i, '0_LU':'23_LD'].to_list()
            stg3_v2 = dfAll.loc[j, '0_LU':'23_LD'].to_list()
            if i == j:
                npDist_stg1[i, j] = 0
                npDist_stg2[i, j] = 0
                npDist_stg3[i, j] = 0
            elif i < j:
                npDist_stg1[i, j] = np.sqrt(sum((float(stg1_v1[k]) - float(stg1_v2[k])) ** 2 for k in range(len(stg1_v1))))
                npDist_stg2[i, j] = np.sqrt(sum((float(stg2_v1[k]) - float(stg2_v2[k])) ** 2 for k in range(len(stg2_v1))))
                npDist_stg3[i, j] = np.sqrt(sum((float(stg3_v1[k]) - float(stg3_v2[k])) ** 2 for k in range(len(stg3_v1))))
            else:
                npDist_stg1[i, j] = npDist_stg1[j, i]
                npDist_stg2[i, j] = npDist_stg2[j, i]
                npDist_stg3[i, j] = npDist_stg3[j, i]
    dates = dfAll['Date'].tolist()
    dfDist_stg1 = pd.DataFrame(npDist_stg1, columns=dates)
    dfDist_stg2 = pd.DataFrame(npDist_stg2, columns=dates)
    dfDist_stg3 = pd.DataFrame(npDist_stg3, columns=dates)
    dfDist_stg1.insert(0, 'Date', dates)
    dfDist_stg2.insert(0, 'Date', dates)
    dfDist_stg3.insert(0, 'Date', dates)
    dfDist_stg1.to_csv(f"{cfg.pathData}distances_stg1.csv", index=False)
    dfDist_stg2.to_csv(f"{cfg.pathData}distances_stg2.csv", index=False)
    dfDist_stg3.to_csv(f"{cfg.pathData}distances_stg3.csv", index=False)
    return dfDist_stg1, npDist_stg1, dfDist_stg2, npDist_stg2, dfDist_stg3, npDist_stg3

if cfg.separate_dist == 1:
    if fn.fileExists(f"{cfg.pathData}distances_stg1.csv") and fn.fileExists(f"{cfg.pathData}distances_stg2.csv") and fn.fileExists(f"{cfg.pathData}distances_stg3.csv"):
        dfDist_stg1 = pd.read_csv(f"{cfg.pathData}distances_stg1.csv")
        dfDist_stg2 = pd.read_csv(f"{cfg.pathData}distances_stg2.csv")
        dfDist_stg3 = pd.read_csv(f"{cfg.pathData}distances_stg3.csv")
        npDist_stg1 = dfDist_stg1.to_numpy()
        npDist_stg2 = dfDist_stg2.to_numpy()
        npDist_stg3 = dfDist_stg3.to_numpy()
        npDist_stg1 = np.delete(npDist_stg1, obj=0, axis=1)
        npDist_stg2 = np.delete(npDist_stg2, obj=0, axis=1)
        npDist_stg3 = np.delete(npDist_stg3, obj=0, axis=1)

    else:    
        dfDist_stg1, npDist_stg1, dfDist_stg2, npDist_stg2, dfDist_stg3, npDist_stg3 = getDistances()

else:
    if fn.fileExists(f"{cfg.pathData}distances.csv"):
        dfDist = pd.read_csv(f"{cfg.pathData}distances.csv")
        npDist = dfDist.to_numpy()
        npDist = np.delete(npDist, obj=0, axis=1)
        npDist_stg3 = npDist
        npDist_stg2 = npDist
        npDist_stg1 = npDist
    else:
        dfDist, npDist = getDistance()
        npDist_stg3 = npDist
        npDist_stg2 = npDist
        npDist_stg1 = npDist


dfMean = pd.read_csv(f"{cfg.pathData}Mean.csv")
dfMeanT = dfMean.transpose()
dfMeanT.columns = [*dfMeanT.iloc[0]]
dfMeanT= dfMeanT.iloc[1:]
dfMeanT.reset_index(inplace=True)


for iter in range(1000):
    logger.info("------------------------------------")
    logger.info(f"Seed {iter}")
    # Cluster to stage 3
    logger.info(f"Clustering for stage 3")
    kmedoids_stg3 = KMedoids(n_clusters=stg3clusters, init=cfg.init, random_state=iter, metric='precomputed', max_iter=100000, method='pam').fit(npDist_stg3)
    npProb = np.array([np.sum(kmedoids_stg3.labels_ == i) / len(kmedoids_stg3.labels_) for i in range(cfg.stg3clusters)])
    npDist_stg2 = npDist_stg3[kmedoids_stg3.medoid_indices_][:, kmedoids_stg3.medoid_indices_]

    npLate = npAll[kmedoids_stg3.medoid_indices_.tolist()][:, 72:]
    npLU = npAll[kmedoids_stg3.medoid_indices_.tolist()][:, 72:96]
    npLD = npAll[kmedoids_stg3.medoid_indices_.tolist()][:, 96:120]

    # Cluster to stage 2
    logger.info(f"Clustering for stage 2")
    kmedoids_stg2 = KMedoids(n_clusters=stg2clusters, init=cfg.init, random_state=iter, metric='precomputed', max_iter=100000, method='pam').fit(npDist_stg2)
    npDist_stg1 = npDist_stg2[kmedoids_stg2.medoid_indices_][:, kmedoids_stg2.medoid_indices_]
    lst = []
    for i in kmedoids_stg2.labels_.tolist():
        j = (kmedoids_stg2.medoid_indices_.tolist())[i]
        lst.append((kmedoids_stg3.medoid_indices_.tolist())[j])
    npSpot = npAll[lst][:, 48:72]
    npGroup = kmedoids_stg2.labels_.tolist()

    # Cluster to stage 1
    logger.info(f"Clustering for stage 1")
    kmedoids_stg1 = KMedoids(n_clusters=stg1clusters, init=cfg.init, random_state=iter, metric='precomputed', max_iter=0, method='pam').fit(npDist_stg1)
    idx = kmedoids_stg3.medoid_indices_[kmedoids_stg2.medoid_indices_[kmedoids_stg1.medoid_indices_[0]]]
    npEarly = npAll[[idx for i in range(cfg.stg3clusters)]][:, 0:48]
    npEU = npAll[[idx for i in range(cfg.stg3clusters)]][:, 0:24]
    npED = npAll[[idx for i in range(cfg.stg3clusters)]][:, 24:48]
        
    # np arrays to DFs
    dfPricesEarly = pd.DataFrame(npEarly, index=None)
    dfPricesSpot = pd.DataFrame(npSpot, index=None)
    dfPricesLate = pd.DataFrame(npLate, index=None)
    dfProbabilities = pd.DataFrame(npProb, index=None)
    dfGroup = pd.DataFrame(npGroup, index=None)
    # PriceData 
    cols = ["EU", "ED", "Spot", "LU", "LD"]
    prob = dfProbabilities.to_numpy().ravel()
    pEU = (dfPricesEarly.to_numpy())[:][:, 0:cfg.hourCount]
    pED = (dfPricesEarly.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount]
    pSpot = dfPricesSpot.to_numpy()
    pLU = (dfPricesLate.to_numpy())[:][:, 0:cfg.hourCount]
    pLD = (dfPricesLate.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount]
    pList = [pEU, pED, pSpot, pLU, pLD]
    column = f'{configuration} {iter}'

    avg = [np.average(np.average(price, axis=0, weights=prob)) for price in pList]

    dfMean_iter = pd.DataFrame([avg], columns=cols)
    dfMean_iter.insert(loc=0, column="index", value=column)
    dfMeanT = pd.concat([dfMeanT, dfMean_iter])
    dfMeanT.to_csv(f"{cfg.pathData}MeansTest.csv")
