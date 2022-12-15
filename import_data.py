from __main__ import (
    pd,
    np,
    read_csv_bitcoinity,
    read_csv_blockchaincom,
)
from os.path import isfile
from sklearn.preprocessing import MinMaxScaler


btc_logreturn = pd.read_csv(
    "Data/Market/btc_logreturns_no_outliers.csv",
    index_col=0,
    parse_dates=True,
    infer_datetime_format=True,
)


btc_volume = read_csv_bitcoinity(
    "Data/Market/bitcoinity_data_btc_price_usd_aggregate.csv"
)
btc_volume = btc_volume.rename(columns={"volume": "Volume"})
btc_volume = btc_volume.drop(columns="price")
btc_volume = btc_volume.fillna(method="ffill")
sentiments = pd.read_csv(
    "Data/Sentiments/sentiments.csv", parse_dates=True, index_col=0
)
sentiments = sentiments.fillna(0)
# handle NA?

sentiments_sd = pd.read_csv(
    "Data/Sentiments/sentiments_sd.csv", parse_dates=True, index_col=0
)
sentiments_sd = sentiments_sd.fillna(0)
blockchain_blocksize = read_csv_bitcoinity(
    "Data/Blockchain/bitcoinity_data_btc_dailyaverageblocksize.csv"
)
blockchain_blocksize = blockchain_blocksize.rename(columns={"Unnamed: 1": "Blocksize"})
blockchain_blocksize = blockchain_blocksize.fillna(method="ffill")
blockchain_transactions = read_csv_bitcoinity(
    "Data/Blockchain/bitcoinity_data_btc_dailytransactionsonnetwork.csv"
)
blockchain_transactions = blockchain_transactions.rename(
    columns={"Unnamed: 1": "Transactions"}
)
blockchain_transactions = blockchain_transactions.fillna(method="ffill")
blockchain_difficulty = read_csv_bitcoinity(
    "Data/Blockchain/bitcoinity_data_btc_miningdifficulty.csv"
)
blockchain_difficulty = blockchain_difficulty.rename(
    columns={"Unnamed: 1": "Difficulty"}
)
blockchain_difficulty = blockchain_difficulty.fillna(method="ffill")
blockchain_hashrate = read_csv_blockchaincom(
    "Data/Blockchain/blockchaincom_btc_hashrate.csv"
)
blockchain_hashrate = blockchain_hashrate.rename(columns={"y": "Hashrate"})
blockchain_hashrate = blockchain_hashrate.fillna(method="ffill")

data_raw = {
    "btc_logreturn": np.array(btc_logreturn["Logreturn"]).reshape(-1, 1),
    "btc_volume": np.array(btc_volume["Volume"]).reshape(-1, 1),
    "blockchain_blocksize": np.array(blockchain_blocksize["Blocksize"]).reshape(-1, 1),
    "blockchain_transactions": np.array(
        blockchain_transactions["Transactions"]
    ).reshape(-1, 1),
    "blockchain_difficulty": np.array(blockchain_difficulty["Difficulty"]).reshape(
        -1, 1
    ),
    "blockchain_hashrate": np.array(blockchain_hashrate["Hashrate"]).reshape(-1, 1),
}
for key in ["btc", "#BTC", "ethereum", "eth", "bitcoin"]:
    data_raw["sentiments_" + key] = np.array(sentiments[key]).reshape(-1, 1)
    data_raw["sentiments_sd_" + key] = np.array(sentiments_sd[key]).reshape(-1, 1)
data_scalers = {key: MinMaxScaler(feature_range=(0, 1)) for key in data_raw.keys()}
data_scaled = {
    key: data_scalers[key].fit_transform(value) for key, value in data_raw.items()
}
