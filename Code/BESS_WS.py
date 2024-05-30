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
import datetime as dt

# from config import *
import config as cfg
import functions as fn

# Setup logging and start a logging instance
cfg.setup_logging()
logger = logging.getLogger(__name__)


def WS(config=cfg.currentConfig):
    pathConfig = f"{cfg.pathData}{config}/"
    logger.info(f"Running configuration '{pathConfig}'")
    # For documentation, see
    # https://www.gurobi.com/documentation/11.0/refman/py_model.html

    logger.info(f"Importing .csv files")
    # Import probabilities
    dfProbabilities = pd.read_csv(f"{pathConfig}Probabilities.csv")
    # Set probabilities to Numpy Array
    prob = dfProbabilities.to_numpy().ravel()

    # Import prices
    dfPricesEarly = pd.read_csv(f"{pathConfig}PricesEarly.csv")
    dfPricesSpot = pd.read_csv(f"{pathConfig}PricesSpot.csv")
    dfPricesLate = pd.read_csv(f"{pathConfig}PricesLate.csv")
    # Set prices to Numpy Array
    pEU = (dfPricesEarly.to_numpy())[:][:, 0:cfg.hourCount] * cfg.scalarEarly
    pED = (dfPricesEarly.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount] * cfg.scalarEarly
    pSpot = dfPricesSpot.to_numpy() * cfg.scalarSpot
    pLU = (dfPricesLate.to_numpy())[:][:, 0:cfg.hourCount] * cfg.scalarLate
    pLD = (dfPricesLate.to_numpy())[:][:, cfg.hourCount : 2 * cfg.hourCount] * cfg.scalarLate


    # Set up environment
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    # Start model
    m = Model(name="BESS")
    m.Params.LogToConsole = 0

    ##########################
    # VARIABLES
    ##########################
    logger.info(f"Creating variables")
    # https://www.gurobi.com/documentation/11.0/refman/py_model_addvars.html
    
    # Battery Charge Level
    bC = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(25)], lb=0, ub=cfg.B_S, vtype=GRB.CONTINUOUS, name='bC')

    # D-1 Early
    qEU = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_P, vtype=GRB.CONTINUOUS, name="qEU")
    qED = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_P, vtype=GRB.CONTINUOUS, name="qED")

    # Spot
    qSpot = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=-cfg.B_S, ub=cfg.B_S, vtype=GRB.CONTINUOUS, name='qSpot')
    qSpotP = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_S, vtype=GRB.CONTINUOUS, name='qSpot+')
    qSpotM = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_S, vtype=GRB.CONTINUOUS, name='qSpot-')

    # D-1 Late
    qLU = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_P, vtype=GRB.CONTINUOUS, name="qLU")
    qLD = m.addVars([s for s in range(cfg.scenarioCount)], [h for h in range(cfg.hourCount)], lb=0, ub=cfg.B_P, vtype=GRB.CONTINUOUS, name="qLD")

    ##########################
    # OBJECTIVE FUNCTION
    ##########################
    # https://gurobi.com/documentation/11.0/refman/py_model_setobjective.html
    logger.info(f"Setting objective function")
    m.setObjective(expr=(
            gp.quicksum(
                1/cfg.scenarioCount * (pEU[s, h] * qEU[s, h] + pED[s, h] * qED[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))
            + gp.quicksum(prob[s] * (pSpot[s, h] * qSpot[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))
            + gp.quicksum(prob[s] * (pLU[s, h] * qLU[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))
            + gp.quicksum(prob[s] * (pLD[s, h] * qLD[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))
    ),
        sense=GRB.MAXIMIZE
    )

    ##########################
    # CONSTRAINTS
    ##########################
    # https://www.gurobi.com/documentation/11.0/refman/py_model_addconstrs.html
    logger.info(f"Setting constraints")
    # Charge level
    m.addConstrs(bC[s, 0] - cfg.B_C == 0 for s in range(cfg.scenarioCount))  # Initial state
    m.addConstrs(bC[s, cfg.hourCount] - cfg.B_C == 0 for s in range(cfg.scenarioCount))  # End state
    m.addConstrs(bC[s, h + 1] == bC[s, h] - qSpot[s, h] + cfg.fcrActivation * (qED[s, h] + qLD[s, h] - qEU[s, h] - qLU[s, h])
                 for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))  # Charge level during operation
    m.addConstrs(qSpot[s, h] == qSpotP[s, h] - qSpotM[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    # D-1 Early
    m.addConstrs(qEU[s, h] <= cfg.B_P * cfg.C_LER for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qED[s, h] <= cfg.B_P * cfg.C_LER for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qEU[s, h] * cfg.C_AT <= bC[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qED[s, h] * cfg.C_AT <= cfg.B_S - bC[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    # Spot
    m.addConstrs(qSpot[s, h] == qSpotP[s, h] - qSpotM[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    m.addConstrs(qSpotP[s, h] + qEU[s, h] * cfg.C_AT <= bC[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qSpotM[s, h] + qED[s, h] * cfg.C_AT <= cfg.B_S - bC[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    m.addConstrs(qSpotP[s, h] + qEU[s, h] <= cfg.B_P for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qSpotM[s, h] + qED[s, h] <= cfg.B_P for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    # D-1 Late    
    m.addConstrs(qEU[s, h] + qLU[s, h] <= cfg.B_P * cfg.C_LER for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qED[s, h] + qLD[s, h] <= cfg.B_P * cfg.C_LER for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))

    m.addConstrs(
        qSpotP[s, h] + (qEU[s, h] + qLU[s, h]) * cfg.C_AT <= bC[s, h] for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qSpotM[s, h] + (qED[s, h] + qLD[s, h]) * cfg.C_AT <= cfg.B_S - bC[s, h] for h in range(cfg.hourCount) for s in
                 range(cfg.scenarioCount))

    m.addConstrs(qSpotP[s, h] + qEU[s, h] + qLU[s, h] <= cfg.B_P for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qSpotM[s, h] + qED[s, h] + qLD[s, h] <= cfg.B_P for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))


    m.addConstrs(qEU[s, h] <= cfg.stg1limit for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qED[s, h] <= cfg.stg1limit for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qLU[s, h] <= cfg.stg3limit for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))
    m.addConstrs(qLD[s, h] <= cfg.stg3limit for h in range(cfg.hourCount) for s in range(cfg.scenarioCount))


    # Non-anticipativity constraints
    # Not applicable for Wait-and-see value

                    

    ##########################
    # OPTIMIZING
    ##########################
    # https://www.gurobi.com/documentation/11.0/refman/py_model_optimize.html
    logger.info(f"Updating")
    m.update()
    logger.info(f"Optimizing")
    m.optimize()
    objV = m.ObjVal

    profitEU = gp.quicksum(1/cfg.scenarioCount * (pEU[s, h] * qEU[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount)).getValue()
    profitED = gp.quicksum(1/cfg.scenarioCount * (pED[s, h] * qED[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount)).getValue()
    profitSpot = gp.quicksum(prob[s] * (pSpot[s, h] * qSpot[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount)).getValue()
    profitLU = (gp.quicksum(prob[s] * (pLU[s, h] * qLU[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))).getValue()
    profitLD = (gp.quicksum(prob[s] * (pLD[s, h] * qLD[s, h]) for s in range(cfg.scenarioCount) for h in range(cfg.hourCount))).getValue()

    return objV, profitEU, profitED, profitSpot, profitLU, profitLD


def runWS(config=cfg.currentConfig):
    startWS = dt.datetime.now()
    objV, profitEU, profitED, profitSpot, profitLU, profitLD = WS(config)
    endWS = dt.datetime.now()
    logger.info("------------------------------------")
    logger.info(f"EUR: {round(objV, 2)}")

    if fn.fileExists(cfg.pathJSON):
        with open(f"{cfg.pathJSON}") as f:
            data = json.load(f)
    else:
        data = {}

    if config not in data:
        data[config] = {}

    data[config]["WS"] = round(objV, 2)
    data[config]["EU_WS"] = round(profitEU, 2)
    data[config]["ED_WS"] = round(profitED, 2)
    data[config]["Spot_WS"] = round(profitSpot, 2)
    data[config]["LU_WS"] = round(profitLU, 2)
    data[config]["LD_WS"] = round(profitLD, 2)
    data[config]["RuntimeWS"] = round((endWS-startWS).total_seconds(),2)
    with open(f"{cfg.pathJSON}", "w") as f:
        json.dump(data, f, indent="    ")
