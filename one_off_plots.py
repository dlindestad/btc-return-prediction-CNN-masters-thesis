import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# used to generate one off plots, mainly for data
# description and visualization
# requires SciencePlots by garrettj403 (github)
# pip install SciencePlots
plt.style.use("science")
# plt.style.use("seaborn-v0_8-whitegrid")


start_date = pd.to_datetime("2012-01-01").date()
end_date = pd.to_datetime("2022-09-01").date()
daterange = pd.date_range(start=start_date, end=end_date).date


def default_plot_style(df, ps: dict):
    if "figsize" in ps.keys():
        plt.figure(figsize=(ps["figsize"]))
    else:
        plt.figure(figsize=(7.76, 1.8))
        # plt.figure(figsize=(7.76, 4.8))
    plt.plot(df)
    plt.title(ps["Title"])
    plt.xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    plt.ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    if "legendpos" in ps.keys():
        plt.legend(ps["legend"], loc=ps["legendpos"], framealpha=1, frameon=True)
    else:
        plt.legend(ps["legend"], loc="best", framealpha=1, frameon=True)
    if ps["logscale"]:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    # if "ticklabel_format" in ps.keys():
    #     plt.ticklabel_format(axis="y", style=ps["ticklabel_format"])
    plt.grid()
    plt.savefig(f"Plots/{ps['figname']}.pdf")
    plt.show()
    return


def default_plot_style_scatter(df, ps: dict):
    plt.figure(figsize=(7.76, 4.8))
    plt.plot(df, ".", markersize=2)
    plt.title(ps["Title"])
    plt.xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    plt.ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    if "legendpos" in ps.keys():
        plt.legend(ps["legend"], loc=ps["legendpos"], framealpha=1, frameon=True)
    else:
        plt.legend(ps["legend"], loc="best", framealpha=1, frameon=True)
    if ps["logscale"]:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.grid()
    plt.savefig(f"Plots/{ps['figname']}.pdf")
    plt.show()
    return


def read_csv_bitcoinity(f_name: str):
    df = pd.read_csv(
        f_name,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )
    # YYYY-MM-DD HH:MM:SS+HH:MM to YYYY-MM-DD
    df.index = pd.to_datetime(df.index).date
    df = df.sort_index()

    # drop dates before start_date and after end_date
    df = df[~(df.index < start_date)]
    df = df[~(df.index > end_date)]

    # insert missing dates with NaN if any
    if len(df.index) != len(daterange):
        print(
            f'\nWarning: Dataset: "{f_name}" missing {len(daterange)-len(df.index)} days, inserting NaN for missing values.',
            "\nMissing dates (only first five are shown):",
        )
        collumn_dict_nan = {collumn_name: np.nan for collumn_name in df.columns}
        for i, date in enumerate(np.sort(list(set(daterange).difference(df.index)))):
            df = pd.concat([df, pd.DataFrame(collumn_dict_nan, index=[date])])
            if i < 5:
                print(date)
        print("\n")
        df = df.sort_index()

    assert len(df.index) == len(daterange)
    return df


def read_csv_blockchaincom(f_name: str):
    df = pd.read_csv(
        f_name,
        index_col="x",
        parse_dates=True,
        infer_datetime_format=True,
    )

    df.index = pd.to_datetime(df.index, unit="ms").date
    df.index.name = "Date"

    # drop dates before start_date and after end_date
    df = df[~(df.index < start_date)]
    df = df[~(df.index > end_date)]

    # insert missing dates with NaN if any
    if len(df.index) != len(daterange):
        print(
            f'\nWarning: Dataset: "{f_name}" missing {len(daterange)-len(df.index)} days, inserting NaN for missing values.',
            "\nMissing dates (only first five are shown):",
        )
        collumn_dict_nan = {collumn_name: np.nan for collumn_name in df.columns}
        for i, date in enumerate(np.sort(list(set(daterange).difference(df.index)))):
            df = pd.concat([df, pd.DataFrame(collumn_dict_nan, index=[date])])
            if i < 5:
                print(date)
        print("\n")
        df = df.sort_index()

    assert len(df.index) == len(daterange)
    return df


def plot_btc_price_bitstamp():
    # BTC price (USD) bitstamp
    f = "Data/Market/bitcoinity_data_btc_price_usd_all_exchanges.csv"
    ps = {
        "Title": "Daily BTC price (USD) - Bitstamp",
        "xlabel": "Date",
        "ylabel": "USD",
        "fontsize": 18,
        "legend": ["BTC price (USD)"],
        "logscale": True,
        "figname": "btc_price_bitstamp",
    }

    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"bitstamp": "Price"})
    for column in df.columns:
        if column != "Price":
            df = df.drop(columns=column)

    default_plot_style(df, ps)
    print(df.head())
    return


def plot_btc_price_aggregate():
    # BTC price (USD) bitstamp
    f = "Data/Market/bitcoinity_data_btc_price_usd_aggregate.csv"
    ps = {
        "Title": "Daily BTC price (USD) - Aggregate of 40 exchanges",
        "xlabel": "Date",
        "ylabel": "USD",
        "fontsize": 18,
        "legend": ["BTC price (USD)"],
        "logscale": True,
        "figname": "btc_price_aggregate",
    }

    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"price": "Price"})
    df = df.drop(columns="volume")

    plt.figure(figsize=(16, 8))
    plt.plot(df)
    plt.title(ps["Title"])
    plt.xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    plt.ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    plt.legend(ps["legend"], loc="lower right")
    if ps["logscale"]:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.show()
    return


def plot_btc_price_major_aggr():
    # BTC price (USD) bitstamp
    f = "Data/Market/bitcoinity_data_btc_price_usd_all_exchanges.csv"
    ps = {
        "Title": "Daily BTC price (USD) - Major exchanges average",
        "xlabel": "Date",
        "ylabel": "USD",
        "fontsize": 18,
        "legend": ["BTC price (USD)"],
        "logscale": True,
        "figname": "btc_price_major_aggr",
    }
    exchange_list = ["bitstamp", "coinbase", "gemini", "kraken"]
    df = read_csv_bitcoinity(f)
    for column in df.columns:
        if column not in exchange_list:
            df = df.drop(columns=column)

    df["Mean"] = df.mean(axis=1)
    default_plot_style(df, ps)
    print(df.head())
    return


def plot_btc_price_major_aggr_difference():
    # BTC price (USD) bitstamp
    f = "Data/Market/bitcoinity_data_btc_price_usd_all_exchanges.csv"
    ps = {
        "Title": "Daily BTC price (USD) - Major exchanges average",
        "xlabel": "Date",
        "ylabel": "% difference",
        "fontsize": 18,
        "legend": ["bitstamp", "coinbase", "gemini", "kraken"],
        "logscale": False,
        "figname": "btc_price_major_aggr_difference",
    }
    exchange_list = ["bitstamp", "coinbase", "gemini", "kraken"]
    df = read_csv_bitcoinity(f)
    for column in df.columns:
        if column not in exchange_list:
            df = df.drop(columns=column)

    df["Mean"] = df.mean(axis=1)

    df_original_diff = df.copy()
    for exchange in exchange_list:
        df_original_diff[exchange + "_diff"] = (
            (df_original_diff[exchange] - df_original_diff["Mean"])
            / df_original_diff["Mean"]
        ) * 100

        df_original_diff = df_original_diff.drop(columns=exchange)
    df_original_diff = df_original_diff.drop(columns="Mean")

    df_clean = df.copy()
    df_clean = df_clean.drop(columns="Mean")
    for exchange in exchange_list:
        for idx, row in df_original_diff.iterrows():
            if row[exchange + "_diff"] >= 2:
                df_clean.loc[idx, exchange] = np.nan

    df_clean["Mean"] = df_clean.mean(axis=1)
    print(df_clean.head())

    return


def plot_btc_price_aggregate_vs_bitstamp():
    # BTC price (USD) bitstamp vs aggregate
    f = "Data/Market/bitcoinity_data_btc_price_usd_aggregate.csv"
    ps = {
        "Title": "Daily BTC price (USD) difference between Bitstamp and aggregate",
        "xlabel": "Date",
        "ylabel": "USD difference",
        "fontsize": 18,
        "legend": ["BTC price (USD) difference"],
        "logscale": False,
        "figname": "btc_price_aggregate_vs_bitstamp",
    }

    df_aggregate = read_csv_bitcoinity(f)
    df_aggregate = df_aggregate.rename(columns={"price": "Price"})
    df_aggregate = df_aggregate.drop(columns="volume")

    # BTC price (USD) bitstamp
    f = "Data/Market/bitcoinity_data_btc_price_usd_bitstamp.csv"
    df_bitstamp = read_csv_bitcoinity(f)
    df_bitstamp = df_bitstamp.rename(columns={"price": "Price"})
    df_bitstamp = df_bitstamp.drop(columns="volume")

    df_aggregate["Difference"] = df_aggregate["Price"] - df_bitstamp["Price"]
    df_aggregate["prcnt_difference"] = (
        (df_aggregate["Price"] - df_bitstamp["Price"]) / df_aggregate["Price"]
    ) * 100
    df = df_aggregate
    df = df.drop(columns="Price")

    # absolute difference
    plt.figure(figsize=(16, 8))
    plt.plot(df["Difference"])
    plt.title(ps["Title"])
    plt.xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    plt.ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    plt.legend(ps["legend"], loc="lower right")
    if ps["logscale"]:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.ylim(-100, 100)
    plt.show()

    # % difference
    plt.figure(figsize=(16, 8))
    plt.plot(df["prcnt_difference"])
    plt.title(ps["Title"])
    plt.xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    plt.ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    plt.legend(ps["legend"], loc="lower right")
    if ps["logscale"]:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.ylim(-7.5, 7.5)
    plt.show()
    return


def plot_btc_volume():
    # BTC Volume
    f = "Data/Market/bitcoinity_data_btc_price_usd_aggregate.csv"
    ps = {
        "Title": "Daily Volume of BTC Traded on exchanges in USD",
        "xlabel": "Date",
        "ylabel": "Volume (\$)",
        "fontsize": 18,
        "legend": ["BTC Volume on exchanges"],
        "logscale": False,
        "figname": "btc_volume",
        "figsize": (7.76, 1.8),
    }
    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"volume": "Volume"})
    df = df.drop(columns="price")
    print(df.head())

    default_plot_style(df, ps)
    return


def plot_btc_daily_blocksize():
    # BTC blockchain - average blocksize
    f = "Data/Blockchain/bitcoinity_data_btc_dailyaverageblocksize.csv"
    ps = {
        "Title": "Daily average blocksize - BTC blockchain",
        "xlabel": "Date",
        "ylabel": "Blocksize MB",
        "fontsize": 18,
        "legend": ["Mean daily blocksize"],
        "logscale": True,
        "figname": "btc_daily_blocksize",
        "figsize": (7.76, 1.8),
        "ticklabel_format": "plain",
    }
    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"Unnamed: 1": "Blocksize"})
    df["Blocksize"] = df["Blocksize"] / 1_000_000
    print(df.head())

    default_plot_style(df, ps)
    return


def plot_btc_daily_transactions():
    # BTC blockchain - daily transactions on network
    f = "Data/Blockchain/bitcoinity_data_btc_dailytransactionsonnetwork.csv"
    ps = {
        "Title": "Daily transactions on network - BTC blockchain",
        "xlabel": "Date",
        "ylabel": "No. of TX",
        "fontsize": 18,
        "legend": ["BTC on chain transactions"],
        "logscale": True,
        "figname": "btc_daily_transaction",
    }
    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"Unnamed: 1": "Transactions"})
    print(df.head())

    default_plot_style(df, ps)
    return


def plot_btc_difficulty():
    # BTC blockchain - mining difficulty
    f = "Data/Blockchain/bitcoinity_data_btc_miningdifficulty.csv"
    ps = {
        "Title": "Daily difficulty - BTC blockchain",
        "xlabel": "Date",
        "ylabel": "Difficulty",
        "fontsize": 18,
        "legend": ["BTC mining difficulty"],
        "logscale": True,
        "figname": "btc_difficulty",
    }
    df = read_csv_bitcoinity(f)
    df = df.rename(columns={"Unnamed: 1": "Difficulty"})
    print(df.head())

    default_plot_style(df, ps)
    return


def plot_btc_hashrate():
    # BTC blockchain - network hashrate
    f = "Data/Blockchain/blockchaincom_btc_hashrate.csv"
    ps = {
        "Title": "Network hashrate - BTC blockchain",
        "xlabel": "Date",
        "ylabel": "Hashrate (TH/s)",
        "fontsize": 18,
        "legend": ["BTC network hashrate"],
        "logscale": True,
        "figname": "btc_hashrate",
    }
    df = read_csv_blockchaincom(f)
    df = df.rename(columns={"y": "Hashrate"})
    df = df.fillna(method="ffill")
    print(df.head())

    default_plot_style(df, ps)
    return


def plot_tweetcounts():
    # Daily twwets for searchstrings
    f = "Data/Tweets/tweetcount.csv"
    ps = {
        "Title": "Daily Number of Tweets",
        "xlabel": "",
        "ylabel": "",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "tweetcounts",
        "figsize": (4, 3),
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    print(df.head())
    default_plot_style(df, ps)
    return


def plot_tweetcounts_monthly():
    # Daily twwets for searchstrings
    f = "Data/Tweets/tweetcount.csv"
    ps = {
        "Title": "Daily Number of Tweets",
        "xlabel": "Date",
        "ylabel": "Daily Tweets",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "legendpos": "best",
        "logscale": False,
        "figname": "tweetcounts",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    df["temp"] = df.index
    df_m = df.groupby(pd.Grouper(freq="M"))[
        "btc", "#BTC", "ethereum", "eth", "bitcoin"
    ].sum()
    df_m = df_m[0:-1]
    print(df_m.head())
    default_plot_style(df_m, ps)
    return


def plot_sentiment():
    # Daily mean tweet sentiment
    f = "Data/Sentiments/sentiments.csv"
    ps = {
        "Title": "Daily mean sentiment",
        "xlabel": "Date",
        "ylabel": "Mean sentiment",
        "fontsize": 18,
        "legend": ["btc", "#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    default_plot_style_scatter(df, ps)
    return


def plot_sentiment_sd():
    # Daily twwets for searchstrings
    f = "Data/Sentiments/sentiments_sd.csv"
    ps = {
        "Title": "Daily Mean Sentiment Standard Deviation",
        "xlabel": "Date",
        "ylabel": "Mean sentiment SD",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment_sd",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    default_plot_style_scatter(df, ps)
    return


def plot_sentiment_and_sd_combined():
    # Daily mean tweet sentiment
    f = "Data/Sentiments/sentiments.csv"
    ps = {
        "Title": "Daily mean sentiment",
        "xlabel": "Date",
        "ylabel": "Mean sentiment",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    fig, ax = plt.subplots(2, figsize=(7.76, 4.8))
    ax[0].plot(df)
    ax[0].set_title(ps["Title"])
    ax[0].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[0].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[0].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[0].set_yscale("linear")
    ax[0].grid()

    # Daily twwets for searchstrings
    f = "Data/Sentiments/sentiments_sd.csv"
    ps = {
        "Title": "Daily Sentiment Standard Deviation",
        "xlabel": "Date",
        "ylabel": "Sentiment $\sigma$",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment_sd",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    ax[1].plot(df)
    ax[1].set_title(ps["Title"])
    ax[1].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[1].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[1].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[1].set_yscale("linear")
    ax[1].grid()
    fig.tight_layout()

    fig.savefig(f"Plots/sentiments_sd_combined.pdf")
    fig.show()
    return


def plot_sentiment_vs_sentiment_incl_combined():
    # Daily mean tweet sentiment
    f = "Data/Sentiments/sentiments.csv"
    ps = {
        "Title": "Daily mean sentiment, no neutral",
        "xlabel": "Date",
        "ylabel": "Mean sentiment",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    fig, ax = plt.subplots(2, figsize=(9, 14))
    ax[0].plot(df)
    ax[0].set_title(ps["Title"])
    ax[0].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[0].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[0].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[0].set_yscale("linear")
    ax[0].set_ylim((-0.8, 1))
    ax[0].grid()

    # Daily tweets for searchstrings
    f = "Data/Sentiments/sentiments_incl_neutral.csv"
    ps = {
        "Title": "Daily mean sentiment, including neutral",
        "xlabel": "Date",
        "ylabel": "Mean sentiment",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment_incl",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    ax[1].plot(df)
    ax[1].set_title(ps["Title"])
    ax[1].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[1].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[1].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[1].set_yscale("linear")
    ax[1].set_ylim((-0.8, 1))
    ax[1].grid()
    fig.tight_layout()

    fig.savefig(f"Plots/sentiments_incl.pdf")
    fig.show()
    return


def plot_sentiment_sd_vs_sentiment_sd_incl_combined():
    # Daily mean tweet sentiment
    f = "Data/Sentiments/sentiments_sd.csv"
    ps = {
        "Title": "Daily sentiment $\sigma$, no neutral",
        "xlabel": "Date",
        "ylabel": "$\sigma$ sentiment",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    fig, ax = plt.subplots(2, figsize=(9, 14))
    ax[0].plot(df)
    ax[0].set_title(ps["Title"])
    ax[0].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[0].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[0].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[0].set_yscale("linear")
    ax[0].set_ylim((0, 1))
    ax[0].grid()

    # Daily tweets for searchstrings
    f = "Data/Sentiments/sentiments_incl_neutral_sd.csv"
    ps = {
        "Title": "Daily sentiment $\sigma$, including neutral",
        "xlabel": "Date",
        "ylabel": "$\sigma$ sentiment",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment_incl",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    ax[1].plot(df)
    ax[1].set_title(ps["Title"])
    ax[1].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[1].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[1].legend(
        ps["legend"], loc="lower right", framealpha=1, frameon=True, prop={"size": 6}
    )
    ax[1].set_yscale("linear")
    ax[1].set_ylim((0, 1))
    ax[1].grid()
    fig.tight_layout()

    fig.savefig(f"Plots/sentiments_sd_incl.pdf")
    fig.show()
    return


def plot_btc_logreturn():
    # Daily logreturn
    f = "Data/Market/btc_logreturns_no_outliers.csv"
    ps = {
        "Title": "BTC log returns",
        "xlabel": "Date",
        "ylabel": "Return",
        "fontsize": 18,
        "legend": ["BTC"],
        "logscale": False,
        "figname": "BTC_logreturns",
        "figsize": (7.76, 1.8),
    }
    df = pd.read_csv(
        f,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    default_plot_style(df, ps)
    return


def plot_sentiment_histograms():
    f = "Data/Sentiments/histogram_df.csv"
    ps = {
        "Title": "Histogram of sentiment, excluding polarity==0 | $\{ Polarity \in[-1,1] | Polarity \\neq 0\}$",
        "xlabel": "Polarity",
        "ylabel": "Count",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    fig, ax = plt.subplots(2, figsize=(9, 14))
    ax[0].bar(df.index, df["bands_excl"], width=1 / 100)
    ax[0].set_title(ps["Title"])
    ax[0].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[0].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[0].set_yscale("linear")
    ax[0].grid()

    ps = {
        "Title": "Histogram of sentiment, including polarity==0 | $Polarity \in[-1,1]$",
        "xlabel": "Polarity",
        "ylabel": "Count",
        "fontsize": 18,
        "legend": ["btc", "\#BTC", "ethereum", "eth", "bitcoin"],
        "logscale": False,
        "figname": "sentiment_incl",
    }
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    ax[1].bar(df.index, df["bands_incl"], width=1 / 100)
    ax[1].set_title(ps["Title"])
    ax[1].set_xlabel(ps["xlabel"], fontsize=ps["fontsize"])
    ax[1].set_ylabel(ps["ylabel"], fontsize=ps["fontsize"])
    ax[1].set_yscale("linear")
    ax[1].grid()
    fig.tight_layout()

    fig.savefig(f"Plots/sentiments_histogram.pdf")
    plt.show()
    return


def plot_loss_history_compare_hyperparams():
    for i in range(0, 10, 2):
        df1 = pd.read_csv(f"model_stats/loss_history_hp{i+1}.csv")
        df2 = pd.read_csv(f"model_stats/loss_history_hp{i+2}.csv")
        fig, ax = plt.subplots(1, 2, figsize=(7.76, 4.8))
        num_epochs = df1["num_epochs"][0]
        train_loss = df1["train_loss"]
        val_loss = df1["val_loss"]
        ax[0].plot(np.arange(0, num_epochs), train_loss, label="Training loss")
        ax[0].plot(np.arange(0, num_epochs), val_loss, label="Validation loss")
        ax[0].legend()
        ax[0].set_yscale(value="log")
        ax[0].set_ylim(0.005, 0.06)
        ax[0].set_title(f"Loss - Hyper parameter set {i+1}")
        ax[0].set_xlabel("Epoch", fontsize=12)
        ax[0].grid()
        num_epochs = df2["num_epochs"][0]
        train_loss = df2["train_loss"]
        val_loss = df2["val_loss"]
        ax[1].plot(np.arange(0, num_epochs), train_loss, label="Training loss")
        ax[1].plot(np.arange(0, num_epochs), val_loss, label="Validation loss")
        ax[1].legend()
        ax[1].set_yscale(value="log")
        ax[1].set_ylim(0.005, 0.06)
        ax[1].set_title(f"Loss - Hyper parameter set {i+2}")
        ax[1].set_xlabel("Epoch", fontsize=12)
        ax[1].grid()
        fig.tight_layout()
        fig.savefig(f"Plots/HP_{i+1}-{i+2}_Loss_comparison.pdf")
        plt.show()
    return


def main_plot_split(
    window,
    horizon,
    epochs,
    model,
    x_train,
    num_epochs,
    model_history,
    data,
    data_scalers,
    pred_test,
    tmp_fldr: str,
    plt_style: str,
):
    from matplotlib.cm import get_cmap

    df = pd.read_csv(
        "Data/Market/btc_logreturns_no_outliers.csv",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )

    fig = plt.figure(figsize=(7.76, 4.8), constrained_layout=True)

    fig.plot(np.arange(0, num_epochs), model_history["loss"])
    fig.plot(
        np.arange(0, num_epochs), model_history["val_loss"], label="Validation loss"
    )
    fig.legend()
    fig.set_yscale(value="log")
    fig.set_title("Loss")

    x_full, _ = create_windows_full(
        data, {"btc_logreturn": data["btc_logreturn"]}, window=window, horizon=horizon
    )
    pred_full = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_full))
    ax["TopRight"].plot(df.index[window:], pred_full)
    ax["TopRight"].set_title("Full prediction")
    ax["TopRight"].set_ylim((-0.7, 0.7))

    ax["BotLeft"].plot(df["Logreturn"])
    ax["BotLeft"].set_title("Actual")
    ax["BotLeft"].set_ylim((-0.7, 0.7))

    ax["BotRight"].plot(df["Logreturn"][: x_train.shape[0] + window * 2])
    ax["BotRight"].plot(df.index[x_train.shape[0] + window * 2 :], pred_test)
    ax["BotRight"].set_title("Test prediction")
    ax["BotRight"].set_ylim((-0.7, 0.7))

    gs = ax["Large"].get_gridspec()
    ax["Large"].remove()
    ax["Large"] = fig.add_subplot(gs[:, 2:], projection="3d")
    interval_3d_plot = 5
    if plt_style == "complex":
        data_plot_3d = np.empty(shape=(round(epochs / 5) * len(pred_full), 3))

        for z in range(interval_3d_plot, epochs + interval_3d_plot, interval_3d_plot):
            model.load_weights(tmp_fldr + f"model.{z:04d}.h5")
            pred_line = data_scalers["btc_logreturn"].inverse_transform(
                model.predict(x_full)
            )
            for x, y in enumerate(pred_line):
                data_plot_3d[
                    (round(z / interval_3d_plot) - 1) * len(pred_full) + x, :
                ] = np.array([x, y[0], z])
        ax["Large"].plot_trisurf(
            data_plot_3d[:, 0],
            data_plot_3d[:, 1],
            data_plot_3d[:, 2],
            cmap="viridis",
            edgecolor="none",
        )
        ax["Large"].view_init(elev=33, azim=7, roll=95)
        ax["Large"].set_xlabel("X: Timestep (Date)")
        ax["Large"].set_ylabel("Y: Logreturn")
        ax["Large"].set_zlabel("Z: Epoch")
    elif plt_style == "simple":
        cmap = get_cmap("viridis_r")
        xdim_3d = np.arange(len(df.index[window:]))
        for z in range(interval_3d_plot, epochs + interval_3d_plot, interval_3d_plot):
            c = cmap(z / epochs)
            print(c)
            print((z / epochs + interval_3d_plot))
            try:
                model.load_weights(tmp_fldr + f"model.{z:04d}.h5")
            except FileNotFoundError:
                break
            pred_line = data_scalers["btc_logreturn"].inverse_transform(
                model.predict(x_full)
            )
            ax["Large"].plot(xdim_3d, pred_line, zs=z, zdir="z", c=c)
        ax["Large"].view_init(elev=33, azim=7, roll=95)
        ax["Large"].set_xlabel("X: Timestep (Date)")
        ax["Large"].set_ylabel("Y: Logreturn")
        ax["Large"].set_zlabel("Z: Epoch")
    fig.tight_layout()
    fig.savefig("Plots/main_plot_last.pdf")
    plt.show()
    return


def plot_loss_history_compare_single():

    df1 = pd.read_csv(f"model_stats/loss_history_last_model.csv")
    df2 = pd.read_csv(f"model_stats/loss_history_hp1.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.76, 4.8))
    num_epochs = df1["num_epochs"][0]
    train_loss = df1["train_loss"]
    val_loss = df1["val_loss"]
    ax[0].plot(np.arange(0, num_epochs), train_loss, label="Training loss")
    ax[0].plot(np.arange(0, num_epochs), val_loss, label="Validation loss")
    ax[0].legend()
    ax[0].set_yscale(value="log")
    ax[0].set_ylim(0.02, 0.03)
    ax[0].set_title(f"Loss")
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].grid()
    num_epochs = df2["num_epochs"][0]
    train_loss = df2["train_loss"]
    val_loss = df2["val_loss"]
    ax[1].plot(np.arange(0, num_epochs), train_loss, label="Training loss")
    ax[1].plot(np.arange(0, num_epochs), val_loss, label="Validation loss")
    ax[1].legend()
    ax[1].set_yscale(value="log")
    ax[1].set_ylim(0.005, 0.06)
    ax[1].set_title(f"Loss")
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(f"Plots/single_Loss_comparison.pdf")
    plt.show()
    return


def plot_loss_history_single():
    df1 = pd.read_csv(f"model_stats/loss_history_last_model_mh1.csv")
    fig, ax = plt.subplots(1, 1, figsize=(7.76, 4.8))
    num_epochs = df1["num_epochs"][0]
    train_loss = df1["train_loss"]
    val_loss = df1["val_loss"]
    ax.plot(np.arange(0, num_epochs), train_loss, label="Training loss")
    ax.plot(np.arange(0, num_epochs), val_loss, label="Validation loss")
    ax.legend()
    ax.set_title(f"Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"Plots/single_Loss_1.pdf")
    plt.show()
    return


def main():
    # plot_btc_price_bitstamp()
    # plot_btc_price_aggregate()
    # plot_btc_price_aggregate_vs_bitstamp()
    # plot_btc_daily_blocksize()
    # plot_btc_daily_transactions()
    # plot_btc_difficulty()
    # plot_btc_hashrate()
    # plot_tweetcounts()
    # plot_tweetcounts_monthly()
    # plot_sentiment()
    # plot_sentiment_sd()
    # plot_btc_logreturn()
    # plot_sentiment_and_sd_combined()
    # plot_btc_daily_blocksize()
    # plot_btc_volume()
    # plot_sentiment_vs_sentiment_incl_combined()
    # plot_sentiment_sd_vs_sentiment_sd_incl_combined()
    # plot_sentiment_histograms()
    # plot_loss_history_compare_hyperparams()
    # plot_loss_history_compare_single()
    plot_loss_history_single()
    return


if __name__ == "__main__":
    main()
