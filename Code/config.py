import os
import time
import datetime as dt
import logging

# Model parameters
stg2clusters = 27  # Number of nodes in stage 2
stg3clusters = 730  # stg2clusters**2  # Number of nodes in stage 3
stg1limit = 10
stg3limit = 10
scenarioCount = stg3clusters  # Number of scenarios
separate_dist = 0  # Flag whether distances are calculated separately for each stage
init = "random"
random_state = 1364  # Random initial point for the clustering algorithm
max_iter = 1000  # Default maximum iterations of clustering
hourCount = 24  # Number of hours in a day
B_P = 10  # Battery power rating (MW)
B_S = 12  # Battery energy rating (MWh)
C_LER = 0.8  # Limited Energy Reserve constant
C_AT = 1 / 3  # Activation time constant
B_C = 5  # Initial charge level constant
if init == "heuristic":
    currentConfig = f"1-{stg2clusters}-{stg3clusters}"
else:
    currentConfig = f"1-{stg2clusters}-{stg3clusters} {random_state}"

scalarEarly = 1  # Scale up or down prices in the Early auction
scalarSpot = 1  # Scale up or down prices in the Spot auction
scalarLate = 1  # Scale up or down prices in the Late auction
fcrActivation = 0  # 0.2  # Amount of the reserved MWh which is spent being activated

sigfigs = 4

# Dates
start = "2022-01-01T00:00"
# end = (dt.datetime.now() + dt.timedelta(days=5)).strftime("%Y-%m-%dT00:00")
end = "2024-01-01T00:00"
start2 = (
    dt.datetime.strptime(start, "%Y-%m-%dT%H:%M") + dt.timedelta(hours=0)
).strftime("%Y-%m-%dT%H:%M")


# Constants
pathCode = "Code/"
pathData = "Data/"
pathCurrentConfig = f"{pathData}{currentConfig}/"
log_dir = "Logs/"
pathJSON = f"{pathData}results.json"

# URL variables
sort = "sort=HourUTC ASC"
priceArea = '"PriceArea":["DK2"]'
auctionType = '"AuctionType":["D-1 early","D-1 late"]'
productName = '"ProductName":["FCR-D ned","FCR-d upp"]'
baseURL = "https://api.energidataservice.dk/dataset/"
fcrURL = f"{baseURL}FcrNdDK2?offset=0&start={start}&end={end}&filter={{{priceArea},{auctionType},{productName}}}&{sort}"
spotURL = f"{baseURL}Elspotprices?offset=0&start={start}&end={end}&filter={{{priceArea}}}&{sort}"


status_code = {
    200: "OK",
    204: "No content",
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    405: "Method not allowed",
    504: "Gateway timeout",
}


def ts():
    return dt.datetime.now().strftime(f"%H:%M:%S.%f")


def setup_logging():
    # Create logs folder if not exists
    pathLogs = "Logs/"
    if not os.path.exists(pathLogs):
        os.makedirs(pathLogs)

    # Get the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M")

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{pathLogs}/{timestamp}.log", encoding="utf-8"),
        ],
    )
