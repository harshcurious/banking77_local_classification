import marimo

__generated_with = "0.25.1"
app = marimo.App(auto_download=["ipynb", "html"])


@app.cell
def _():
    import altair as alt
    alt.data_transformers.enable("vegafusion")

    import marimo as mo

    return alt, mo


@app.cell
def _():
    import json
    import urllib.request

    _info = json.loads(
        urllib.request.urlopen(
            "https://huggingface.co/api/datasets/PolyAI/banking77"
        ).read()
    )
    print([s["rfilename"] for s in _info["siblings"]])
    return json, urllib


@app.cell
def _(json, urllib):
    import polars as pl
    from pathlib import Path

    _base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

    data_dir = Path("data")  # relative to the working directory
    parquet_dir = data_dir / "banking77"
    train_path = parquet_dir / "train.parquet"
    test_path = parquet_dir / "test.parquet"
    categories_path = parquet_dir / "categories.json"

    if train_path.exists() and test_path.exists() and categories_path.exists():
        # Load from the local parquet/json cache
        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)
        _categories = json.loads(categories_path.read_text())
    else:
        # Download from the source and cache locally
        parquet_dir.mkdir(parents=True, exist_ok=True)

        train_df = pl.read_csv(f"{_base}/train.csv")
        test_df = pl.read_csv(f"{_base}/test.csv")
        _categories = json.loads(
            urllib.request.urlopen(f"{_base}/categories.json").read()
        )

        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        categories_path.write_text(json.dumps(_categories))

    train_df.head()
    return parquet_dir, pl, test_df, train_df


@app.cell
def _(parquet_dir):
    sorted(p.name for p in parquet_dir.glob("*.parquet"))
    return


@app.cell
def _(train_df):
    train_df["category"].n_unique()
    return


@app.cell
def _(train_df):
    train_df["category"].value_counts()
    return


@app.cell
def _(pl, train_df):
    train_df.with_columns(
        pl.col("text").str.len_chars().alias("n_chars"),
        pl.col("text").str.count_matches(r"\S+").alias("word_count"),
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Relating `category` to `n_char`

    You have 77 categories and two text columns' worth of statistics, so there are two natural views:

    1. **Distribution view** — a horizontal box plot with one box per category, sorted by median character count. This shows both the typical length *and* the spread within each intent.
    2. **Summary view** — a bar chart of the mean character count per category, sorted, which is easier to read when you just want the ranking.

    First we rebuild the per-row stats (train + test combined) as a tidy dataframe.
    """)
    return


@app.cell
def _(pl, test_df, train_df):
    def _add_text_stats(df: pl.DataFrame, split: str) -> pl.DataFrame:
        return df.with_columns(
            pl.lit(split).alias("split"),
            pl.col("text").str.len_chars().alias("n_char"),
            pl.col("text").str.count_matches(r"\S+").alias("n_word"),
        )


    text_stats_df = pl.concat([
        _add_text_stats(train_df, "train"),
        _add_text_stats(test_df, "test"),
    ])

    text_stats_df.head()
    return (text_stats_df,)


@app.cell
def _(pl, text_stats_df):
    category_char_stats = (
        text_stats_df.group_by("category")
        .agg(
            pl.col("n_char").mean().alias("mean_n_char"),
            pl.col("n_char").median().alias("median_n_char"),
            pl.col("n_char").std().alias("std_n_char"),
            pl.col("n_char").min().alias("min_n_char"),
            pl.col("n_char").max().alias("max_n_char"),
            pl.len().alias("n_rows"),
        )
        .sort("median_n_char")
    )

    category_char_stats.head(10)
    return (category_char_stats,)


@app.cell
def _(alt, category_char_stats, pl, text_stats_df):
    subset_categories = [
        "card_arrival",
        "card_not_working",
        "top_up_failed",
        "cash_withdrawal_charge",
        "apple_pay_or_google_pay",
        "beneficiary_not_allowed",
        "verify_my_identity",
        "unable_to_verify_identity",
        "why_verify_identity",
        "declined_card_payment",
        "pending_card_payment",
        "card_swallowed",
        "lost_or_stolen_card",
        "card_payment_wrong_exchange_rate",
        "exchange_rate",
    ]

    subset_df = text_stats_df.filter(pl.col("category").is_in(subset_categories))

    # order the intents by their median character count (ascending)
    subset_order = (
        category_char_stats.filter(pl.col("category").is_in(subset_categories))
        .sort("median_n_char")["category"]
        .to_list()
    )

    subset_box_chart = (
        alt.Chart(subset_df)
        .mark_boxplot(size=24)
        .encode(
            x=alt.X("n_char:Q", title="characters per message"),
            y=alt.Y("category:N", title=None, sort=subset_order),
            color=alt.Color("category:N", legend=None, scale=alt.Scale(scheme="tableau20")),
            tooltip=[
                alt.Tooltip("category:N", title="intent"),
                alt.Tooltip("n_char:Q", aggregate="count", title="messages", format=","),
                alt.Tooltip("n_char:Q", aggregate="mean", title="mean chars", format=".1f"),
                alt.Tooltip("n_char:Q", aggregate="median", title="median chars", format=".1f"),
            ],
        )
        .properties(
            title="Character length distribution per intent (15 representative categories)",
            width=620,
            height=650,
        )
    )

    subset_box_chart
    return


@app.cell
def _(alt, category_char_stats, text_stats_df):
    all_categories_order = category_char_stats["category"].to_list()

    # Altair's vegafusion data transformer only understands pandas objects; a polars
    # frame silently falls back to the default transformer, which caps datasets at
    # 5000 rows. text_stats_df has ~13k rows, so chart against a pandas copy holding
    # just the two columns the charts below encode.
    text_stats_chart_pd = text_stats_df.select("category", "n_char").to_pandas()

    all_categories_box_chart = (
        alt.Chart(text_stats_chart_pd)
        .mark_boxplot(size=6, color="#4c78a8")
        .encode(
            x=alt.X("n_char:Q", title="characters per message"),
            y=alt.Y("category:N", title=None, sort=all_categories_order),
            tooltip=[
                alt.Tooltip("category:N", title="intent"),
                alt.Tooltip("n_char:Q", aggregate="count", title="messages", format=","),
                alt.Tooltip("n_char:Q", aggregate="median", title="median chars", format=".1f"),
            ],
        )
        .properties(
            title="All 77 intents sorted by median character length",
            width=620,
            height=1600,
        )
    )

    all_categories_box_chart
    return (text_stats_chart_pd,)


@app.cell
def _(alt, category_char_stats, text_stats_chart_pd):
    mean_char_bars = (
        alt.Chart(category_char_stats)
        .mark_bar()
        .encode(
            x=alt.X("mean_n_char:Q", title="mean characters per message"),
            y=alt.Y("category:N", title=None, sort=alt.EncodingSortField(field="mean_n_char", order="ascending")),
            color=alt.Color("mean_n_char:Q", scale=alt.Scale(scheme="oranges"), title="mean\nchars"),
            tooltip=[
                alt.Tooltip("category:N", title="intent"),
                alt.Tooltip("mean_n_char:Q", title="mean chars", format=".1f"),
                alt.Tooltip("median_n_char:Q", title="median chars", format=".1f"),
                alt.Tooltip("std_n_char:Q", title="std dev", format=".1f"),
                alt.Tooltip("n_rows:Q", title="messages", format=","),
            ],
        )
        .properties(
            title="Mean character length per intent (with ±1 std dev)",
            width=620,
            height=1500,
        )
    )

    # error bars are computed directly from the raw rows: mean ± 1 standard deviation
    mean_char_error_bars = (
        alt.Chart(text_stats_chart_pd)
        .mark_errorbar(extent="stdev", color="#333333", ticks=True)
        .encode(
            x=alt.X("n_char:Q", title="mean characters per message"),
            y=alt.Y("category:N", title=None, sort=alt.EncodingSortField(field="mean_n_char", order="ascending")),
        )
    )

    mean_char_bar_chart = mean_char_bars + mean_char_error_bars

    mean_char_bar_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading the plots

    - The spread within a category is usually **larger than** the differences between categories, so `n_char` is a weak signal on its own — but the extremes are informative.
    - Very short-utterance intents (e.g. balance/identity confirmations) sit at the low end of the ranking.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
