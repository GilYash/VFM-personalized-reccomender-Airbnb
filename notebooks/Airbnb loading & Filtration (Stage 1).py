# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Load the Airbnb Data

# COMMAND ----------

storage_account = "lab94290"  
container = "airbnb"

# COMMAND ----------

# MAGIC %md
# MAGIC Reading airbnb data from Parquet.

# COMMAND ----------

sas_token= "sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D"
sas_token = sas_token.lstrip('?')
spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS")
spark.conf.set(f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set(f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", sas_token)

# COMMAND ----------

# MAGIC %md
# MAGIC display the data

# COMMAND ----------

path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/airbnb_1_12_parquet"

airbnb = spark.read.parquet(path)
display(airbnb.limit(5))

print(f"Number of records in the dataframe: {airbnb.count()}")

# COMMAND ----------

# print the structure of the dataframe
airbnb.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filter rows to top K cities with a threshold of N>= X listings 

# COMMAND ----------

# MAGIC %md
# MAGIC create city and country columns from location

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


# COMMAND ----------

airbnb_loc = (
    airbnb
    .withColumn("city", F.trim(F.split(F.col("location"), ",").getItem(0)))
    .withColumn(
        "country",
        F.trim(F.element_at(F.split(F.col("location"), ","), -1))
    )
    .filter(
        F.col("city").isNotNull() &
        F.col("country").isNotNull() &
        F.col("lat").isNotNull() &
        F.col("long").isNotNull()
    )
)


display(airbnb_loc.limit(5))

print(f"Number of records in the dataframe: {airbnb_loc.count()}")

total_rows = airbnb_loc.count()


# COMMAND ----------

city_country_counts = (
    airbnb_loc
    .groupBy("country", "city")
    .agg(F.count("*").alias("listings_in_city"))
)

display(city_country_counts.limit(15))

# COMMAND ----------

city_counts = (
    airbnb_loc
    .groupBy("city")
    .agg(F.count("*").alias("listings_in_country"))
    .orderBy(F.desc("listings_in_country"))
    .limit(50)
)

display(city_counts)

top_30_sum = city_counts.limit(30).agg(F.sum("listings_in_country").alias("top_30_sum")).collect()[0]["top_30_sum"]
top_40_sum = city_counts.limit(40).agg(F.sum("listings_in_country").alias("top_40_sum")).collect()[0]["top_40_sum"]
top_50_sum = city_counts.limit(50).agg(F.sum("listings_in_country").alias("top_50_sum")).collect()[0]["top_50_sum"]

print(f"Sum of listings in top 30 cities: {top_30_sum}")
print(f"Sum of listings in top 40 cities: {top_40_sum}")
print(f"Sum of listings in top 50 cities: {top_50_sum}")

# COMMAND ----------

COUNTRY_THRESHOLD = 4000

country_counts = (
    airbnb_loc
    .groupBy("country")
    .agg(F.count("*").alias("listings_in_country"))
)

eligible_countries = (
    country_counts
    .filter(F.col("listings_in_country") >= COUNTRY_THRESHOLD)
    .orderBy(F.desc("listings_in_country"))
)

display(eligible_countries)

top_countries = eligible_countries.limit(10)
display(top_countries)

total_eligible_listings = eligible_countries.agg(F.sum("listings_in_country").alias("total_eligible_listings")).collect()[0]["total_eligible_listings"]
percentage_eligible = (total_eligible_listings / total_rows) * 100

print(f"Total listings in eligible countries: {total_eligible_listings}")
print(f"Percentage of listings in eligible countries: {percentage_eligible:.2f}%")

us_city_counts = (
    airbnb_loc
    .filter(F.col("country") == "United States")
    .groupBy("city")
    .agg(F.count("*").alias("listings_in_city"))
    .orderBy(F.desc("listings_in_city"))
)

display(us_city_counts)
# for each threshold print the number of listings were left with 
thresholds = list(range(0, us_city_counts.agg(F.max("listings_in_city")).collect()[0][0] + 1000, 1000))
for t in thresholds:
    count = us_city_counts.filter(F.col("listings_in_city") >= t).agg(F.sum("listings_in_city")).collect()[0][0]
    print(f"Listings in US cities with at least {t} listings: {count if count is not None else 0}")

# COMMAND ----------

# MAGIC %md
# MAGIC ~1/3 of the data is in the United States so we will focus on that market solely. With listings above the 1000s threshold leaving us with ~234K listings from almost 100 cities. 

# COMMAND ----------

us_cities_above_1000 = (
    airbnb_loc
    .filter(F.col("country") == "United States")
    .groupBy("city")
    .agg(F.count("*").alias("listings_in_city"))
    .filter(F.col("listings_in_city") > 1000)
    .select("city")
)

us_filtered = (
    airbnb_loc
    .filter(
        (F.col("country") == "United States") &
        (F.col("city").isin([row["city"] for row in us_cities_above_1000.collect()]))
    )
)

display(us_filtered)
print(f"Number of records in the dataframe: {us_filtered.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # remove nulls from price and handle nulls in num of property reviews

# COMMAND ----------

total_count = us_filtered.count()
null_percentages = (
    us_filtered
    .select([
        (F.sum(F.col(c).isNull().cast("int")) / total_count * 100).alias(c)
        for c in us_filtered.columns
    ])
    .toPandas()
    .T
    .reset_index()
    .rename(columns={"index": "column", 0: "null_percentage"})
)

for row in null_percentages.sort_values("null_percentage", ascending=False).itertuples(index=False):
    print(f"{row.column}: {row.null_percentage:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter to only relevant columns (VFM base + listings metadata + geographical data)

# COMMAND ----------

# List of relevant columns to keep; add or remove column names as needed
relevant_columns = [
    # geographic information
    'city', 'country',  'lat', 'long',
    # listings information
    'name', 'property_id', 'final_url', 'listing_title', 'guests', 'pets_allowed', 'is_supperhost', 'is_guest_favorite',
    # vfm information
      'price', 'reviews', 'ratings', 'amenities', 'property_number_of_reviews',
]

us_filtered_vfm = us_filtered.select(*relevant_columns)
display(us_filtered_vfm)

# COMMAND ----------

total_count = us_filtered_vfm.count()
num_columns = len(us_filtered_vfm.columns)
print(f"Shape: ({total_count}, {num_columns})")

null_percentages = (
    us_filtered_vfm
    .select([
        (F.sum(F.col(c).isNull().cast("int")) / total_count * 100).alias(c)
        for c in us_filtered_vfm.columns
    ])
    .toPandas()
    .T
    .reset_index()
    .rename(columns={"index": "column", 0: "null_percentage"})
)

for row in null_percentages.sort_values("null_percentage", ascending=False).itertuples(index=False):
    print(f"{row.column}: {row.null_percentage:.2f}%")

# COMMAND ----------

null_price_by_city = (
    us_filtered
    .groupBy("city")
    .agg(
        F.sum(F.col("price").isNull().cast("int")).alias("null_price_count"),
        F.count("*").alias("total_count")
    )
    .withColumn("null_price_pct", (F.col("null_price_count") / F.col("total_count")) * 100)
    .orderBy(F.desc("null_price_count"))
    .select("city", "null_price_count", "total_count", "null_price_pct")
)

display(null_price_by_city)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Remove NULLs and variable imputations

# COMMAND ----------

from pyspark.sql.functions import when, col, lit, trim, isnull, parse_json, is_variant_null

# Impute 'reviews' column: if null or empty JSON, set to 'no reviews'
us_filtered_vfm = us_filtered_vfm.withColumn(
    "reviews",
    when(
        isnull(col("reviews")) | (trim(col("reviews")) == "") | (trim(col("reviews")) == "{}") | is_variant_null(parse_json(col("reviews"))),
        lit("no reviews")
    ).otherwise(col("reviews"))
)

# Impute 'property_number_of_reviews' only if null: 0 if 'reviews' is 'no reviews', else 1
us_filtered_vfm = us_filtered_vfm.withColumn(
    "property_number_of_reviews",
    when(
        isnull(col("property_number_of_reviews")),
        when(col("reviews") == "no reviews", lit(0)).otherwise(lit(1))
    ).otherwise(col("property_number_of_reviews"))
)

display(us_filtered_vfm)

# COMMAND ----------

# recount nulls to see if the imputation worked
total_count = us_filtered_vfm.count()
num_columns = len(us_filtered_vfm.columns)
print(f"Shape: ({total_count}, {num_columns})")

null_percentages = (
    us_filtered_vfm
    .select([
        (F.sum(F.col(c).isNull().cast("int")) / total_count * 100).alias(c)
        for c in us_filtered_vfm.columns
    ])
    .toPandas()
    .T
    .reset_index()
    .rename(columns={"index": "column", 0: "null_percentage"})
)

for row in null_percentages.sort_values("null_percentage", ascending=False).itertuples(index=False):
    print(f"{row.column}: {row.null_percentage:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Precentile clipping and capping 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

# 1) Parse reviews JSON string -> array<string>
reviews_schema = T.ArrayType(T.StringType())

df = (
    us_filtered_vfm
    .withColumn(
        "reviews_arr",
        F.when(F.col("reviews").isNull(), F.array().cast(reviews_schema))
         .otherwise(F.from_json(F.col("reviews"), reviews_schema))
    )
)

# 2) Count DISTINCT reviews per listing (per row), fill with 0 if reviews_arr is null, and set reviews_arr to ['no reviews'] if null
df = (
    df
    .withColumn(
        "reviews_arr",
        F.when(F.col("reviews_arr").isNull(), F.array(F.lit("no reviews"))).otherwise(F.col("reviews_arr"))
    )
    .withColumn(
        "num_reviews_distinct",
        F.when(F.col("reviews_arr") == F.array(F.lit("no reviews")), F.lit(0))
         .otherwise(F.size(F.array_distinct(F.col("reviews_arr"))))
    )
    .drop("num_reviews_total")
)

us_filtered_vfm_with_counts = df

display(us_filtered_vfm_with_counts.select("reviews_arr"))
display(us_filtered_vfm_with_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Different approach to imputation of Property number of reviews
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute the property number of reviews column with the actual counts of reviews. 
# MAGIC
# MAGIC - If no reviews will write 0
# MAGIC - Seems that due to spark limitations or something else the maximum number of ACTUAL reviews is 24

# COMMAND ----------

from pyspark.sql import functions as F

us_filtered_vfm = (
    us_filtered_vfm_with_counts
    .withColumn(
        "property_number_of_reviews",
        F.when(
            F.col("property_number_of_reviews").isNull()
            | (F.col("property_number_of_reviews") < F.col("num_reviews_distinct")),
            F.col("num_reviews_distinct")
        ).otherwise(F.col("property_number_of_reviews"))
    )
)

display(us_filtered_vfm)


# COMMAND ----------


# --- percentiles to print ---
percentiles = [0.95, 0.975, 0.99]

# ensure numeric + drop nulls for quantiles
dfq = us_filtered_vfm.select(F.col("property_number_of_reviews").cast("double").alias("property_number_of_reviews")) \
                     .where(F.col("property_number_of_reviews").isNotNull())

percentile_values = dfq.approxQuantile("property_number_of_reviews", percentiles, 0.01)

# --- summary stats (Spark-safe median) ---
stats = (
    us_filtered_vfm
    .agg(
        F.expr("percentile_approx(num_reviews_distinct, 0.5)").alias("median_num_reviews_distinct"),
        F.mean("num_reviews_distinct").alias("mean_num_reviews_distinct"),
        F.min("num_reviews_distinct").alias("min_num_reviews_distinct"),
        F.max("num_reviews_distinct").alias("max_num_reviews_distinct"),

        F.expr("percentile_approx(property_number_of_reviews, 0.5)").alias("median_property_number_of_reviews"),
        F.mean("property_number_of_reviews").alias("mean_property_number_of_reviews"),
        F.min("property_number_of_reviews").alias("min_property_number_of_reviews"),
        F.max("property_number_of_reviews").alias("max_property_number_of_reviews")
    )
    .collect()[0]
)

print(f"num_reviews_distinct - Median: {stats['median_num_reviews_distinct']}")
print(f"num_reviews_distinct - Mean: {stats['mean_num_reviews_distinct']}")
print(f"num_reviews_distinct - Min: {stats['min_num_reviews_distinct']}")
print(f"num_reviews_distinct - Max: {stats['max_num_reviews_distinct']}")
print(f"property_number_of_reviews - Median: {stats['median_property_number_of_reviews']}")
print(f"property_number_of_reviews - Mean: {stats['mean_property_number_of_reviews']}")
print(f"property_number_of_reviews - Min: {stats['min_property_number_of_reviews']}")
print(f"property_number_of_reviews - Max: {stats['max_property_number_of_reviews']}")

# --- print percentiles ---
for p, v in zip(percentiles, percentile_values):
    # keep nice formatting: 95th, 97.5th, 99th
    label = f"{p*100:g}th"  # removes trailing .0
    print(f"property_number_of_reviews - {label} percentile: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC **We will cap the property_number_of_reviews variable in the 97.5 precentile mark**

# COMMAND ----------

from pyspark.sql import functions as F

# convert property_number_of_reviews to int
us_filtered_vfm = us_filtered_vfm.withColumn(
    "property_number_of_reviews",
    F.col("property_number_of_reviews").cast("int")
)

# compute p97.5 (with 1% relative error, change if you want tighter)
p975 = us_filtered_vfm.approxQuantile("property_number_of_reviews", [0.975], 0.01)[0]

# cap (winsorize) at p97.5
us_filtered_vfm = us_filtered_vfm.withColumn(
    "property_number_of_reviews_capped",
    F.when(F.col("property_number_of_reviews") > F.lit(p975), F.lit(p975))
     .otherwise(F.col("property_number_of_reviews"))
)

print("Capped property_number_of_reviews at p97.5 =", p975)


# COMMAND ----------

# PySpark histogram BEFORE vs AFTER capping using the same bins + matplotlib
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer
import numpy as np
import matplotlib.pyplot as plt

col_raw = "property_number_of_reviews"
col_capped = "property_number_of_reviews_capped"

# 0) Make sure you kept the "before" values
# If you haven't already, run this BEFORE you cap:
# us_filtered_vfm = us_filtered_vfm.withColumn(col_raw, F.col(col_capped))

# 1) Define common bin edges (splits) from the RAW distribution
#    We use [min .. p99] to avoid extreme tails dominating the bins.
min_v, p99_v = us_filtered_vfm.approxQuantile(col_raw, [0.0, 0.99], 0.001)
min_v = float(min_v)
p99_v = float(p99_v)

n_bins = 50
splits = np.linspace(min_v, p99_v, n_bins + 1).tolist()
# Bucketizer requires -inf/+inf to catch out-of-range values
splits = [-float("inf")] + splits + [float("inf")]

# 2) Bucketize both columns into the SAME bins and count
b_raw = Bucketizer(splits=splits, inputCol=col_raw, outputCol="bin_raw", handleInvalid="keep")
b_cap = Bucketizer(splits=splits, inputCol=col_capped, outputCol="bin_cap", handleInvalid="keep")

tmp = b_cap.transform(b_raw.transform(us_filtered_vfm.select(col_raw, col_capped)))

counts_raw = (
    tmp.groupBy("bin_raw").count()
       .orderBy("bin_raw")
       .collect()
)
counts_cap = (
    tmp.groupBy("bin_cap").count()
       .orderBy("bin_cap")
       .collect()
)

# 3) Convert counts to aligned arrays for plotting (bin index -> count)
# bins are 0..(len(splits)-2)
n = len(splits) - 1
raw_arr = np.zeros(n, dtype=int)
cap_arr = np.zeros(n, dtype=int)

for r in counts_raw:
    raw_arr[int(r["bin_raw"])] = r["count"]
for r in counts_cap:
    cap_arr[int(r["bin_cap"])] = r["count"]

# 4) Build x-axis bin centers for nicer plot (ignore -inf/+inf edges)
finite_edges = np.array(splits[1:-1], dtype=float)  # the linspace edges
centers = (finite_edges[:-1] + finite_edges[1:]) / 2.0
# These correspond to bins 1..n-2 (since bin 0 is -inf..min, last is p99..inf)
# We'll plot only the finite-range bins for readability:
start_bin = 1
end_bin = n - 2

x = centers[start_bin:end_bin]
y_raw = raw_arr[start_bin:end_bin]
y_cap = cap_arr[start_bin:end_bin]

# 5) Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y_raw, label="Before capping")
plt.plot(x, y_cap, label="After capping")
plt.xlabel(col_capped)
plt.ylabel("Count")
plt.title("Histogram (same bins): before vs after capping")
plt.legend()
plt.show()


# COMMAND ----------

max_val = us_filtered_vfm.agg(F.max("property_number_of_reviews_capped")).collect()[0][0]
print(f"Max value of property_number_of_reviews_capped: {max_val}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove nulls from the columns that are left

# COMMAND ----------

total_count = us_filtered_vfm.count()
null_counts = (
    us_filtered_vfm.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in us_filtered_vfm.columns
    ])
    .toPandas()
    .T
    .reset_index()
    .rename(columns={"index": "column", 0: "null_count"})
    .sort_values(by="null_count", ascending=False)
    .reset_index(drop=True)
)
null_counts["null_pct"] = (null_counts["null_count"] / total_count) * 100

display(null_counts)

# COMMAND ----------

# remove nulls 
us_filtered_vfm = us_filtered_vfm.dropna()
display(us_filtered_vfm)
print("Shape:", (us_filtered_vfm.count(), len(us_filtered_vfm.columns)))

# COMMAND ----------

percentiles = [0.95, 0.975, 0.99]

us_filtered_vfm = us_filtered_vfm.withColumn("price", F.col("price").cast("float"))

pvals_reviews = us_filtered_vfm.approxQuantile("property_number_of_reviews_capped", percentiles, 0.01)
pvals_price = us_filtered_vfm.approxQuantile("price", percentiles, 0.01)

for p, v in zip(percentiles, pvals_reviews):
    print(f"property_number_of_reviews_capped - {int(p*100)}th percentile: {v}")
for p, v in zip(percentiles, pvals_price):
    print(f"price - {int(p*100)}th percentile: {v}")

stats_df = us_filtered_vfm.select("property_number_of_reviews_capped", "price").summary()
display(stats_df)

# COMMAND ----------

p975_price = us_filtered_vfm.approxQuantile("price", [0.975], 0.01)[0]
us_filtered_vfm = us_filtered_vfm.filter(F.col("price") <= F.lit(p975_price))
print("Shape:", (us_filtered_vfm.count(), len(us_filtered_vfm.columns)))

# COMMAND ----------

max_price = us_filtered_vfm.agg(F.max("price")).collect()[0][0]
print(f"Max value of price: {max_price}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 EDA on the columns to see the distributions

# COMMAND ----------

us_filtered_vfm.select("price").toPandas().plot.hist(bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create a sentiment column using Spark NLP

# COMMAND ----------

from pyspark.sql import functions as F, types as T
import json, ast, re

# ============================================================
# CONFIG — set these to YOUR df column names
# ============================================================
LISTING_ID_COL = "property_id"     
REVIEWS_ARR_COL = "reviews_arr"   # as in your sample

TOTAL_REVIEWS_COUNT_COL = None    # set to a column name if you have it, else keep None

confidence_saturation_parameter = 17   # k (median of property_number_of_reviews_capped)
max_sentiment_impact = 0.2             # ±10%

# ============================================================
# 1) Robustly parse reviews_arr into array<string> (emoji-safe)
# ============================================================
REV_SCHEMA = T.ArrayType(T.StringType())

def _parse_reviews_any(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(r) for r in x if r is not None]
    if isinstance(x, dict):
        for k in ("reviews", "reviews_arr", "comments", "texts"):
            v = x.get(k)
            if isinstance(v, list):
                return [str(r) for r in v if r is not None]
        return []
    if not isinstance(x, str):
        return []

    s = x.strip()
    if not s:
        return []

    # JSON array
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(r) for r in v if r is not None]
    except Exception:
        pass

    # python literal array
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(r) for r in v if r is not None]
    except Exception:
        pass

    # multiple arrays concatenated in one string
    chunks = re.findall(r"\[[\s\S]*?\]", s)
    out = []
    for c in chunks:
        try:
            v = json.loads(c)
            if isinstance(v, list):
                out.extend([str(r) for r in v if r is not None])
                continue
        except Exception:
            pass
        try:
            v = ast.literal_eval(c)
            if isinstance(v, list):
                out.extend([str(r) for r in v if r is not None])
        except Exception:
            pass
    return out

parse_reviews_udf = F.udf(_parse_reviews_any, REV_SCHEMA)

reviews_dtype = df.schema[REVIEWS_ARR_COL].dataType
if isinstance(reviews_dtype, T.ArrayType):
    df_fixed = df.withColumn("reviews_list", F.col(REVIEWS_ARR_COL).cast(REV_SCHEMA))
else:
    df_fixed = df.withColumn("reviews_list", parse_reviews_udf(F.col(REVIEWS_ARR_COL)))

df_fixed = df_fixed.withColumn(
    "reviews_list",
    F.when(F.col("reviews_list").isNull(), F.array().cast(REV_SCHEMA)).otherwise(F.col("reviews_list"))
)

# light cleanup (keeps emojis, just removes HTML tags / <br/>)
df_fixed = df_fixed.withColumn(
    "reviews_list",
    F.transform(
        F.col("reviews_list"),
        lambda x: F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(x, r"(?i)<br\s*/?>", " "),
                    r"<[^>]+>", " "
                ),
                r"\s+", " "
            )
        )
    )
)

# ============================================================
# 2) Explode to review-level rows
# ============================================================
reviews_long = (
    df_fixed
    .select(
        F.col(LISTING_ID_COL).alias("listing_id"),
        F.explode_outer(F.col("reviews_list")).alias("review_text")
    )
    .filter(F.col("review_text").isNotNull() & (F.length(F.trim(F.col("review_text"))) > 0))
)

# ============================================================
# 3) VADER sentiment via Pandas UDF (fast + scalable)
# ============================================================
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("double")
def vader_compound_udf(texts: pd.Series) -> pd.Series:
    # created per executor / batch; emoji-safe (unicode)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return texts.fillna("").map(lambda t: analyzer.polarity_scores(t)["compound"])

reviews_scored = reviews_long.withColumn(
    "review_sentiment_score",
    vader_compound_udf(F.col("review_text"))
)

# ============================================================
# 4) Aggregate per listing
# ============================================================
listing_sentiment = (
    reviews_scored
    .groupBy("listing_id")
    .agg(
        F.avg("review_sentiment_score").alias("average_sentiment_score"),
        F.count(F.lit(1)).alias("number_of_text_reviews")
    )
)

base_counts = df_fixed.select(
    F.col(LISTING_ID_COL).alias("listing_id"),
    (F.col(TOTAL_REVIEWS_COUNT_COL).cast("int") if TOTAL_REVIEWS_COUNT_COL else F.lit(None).cast("int")).alias("number_of_all_reviews")
)

listing_features = base_counts.join(listing_sentiment, on="listing_id", how="left")

listing_features = (
    listing_features
    .withColumn("average_sentiment_score", F.coalesce(F.col("average_sentiment_score"), F.lit(0.0)))
    .withColumn("number_of_text_reviews", F.coalesce(F.col("number_of_text_reviews"), F.lit(0)))
    .withColumn("number_of_all_reviews", F.coalesce(F.col("number_of_all_reviews"), F.col("number_of_text_reviews")))
)

# ============================================================
# 5) Confidence + multiplier
# ============================================================
listing_features = listing_features.withColumn(
    "confidence_weight",
    F.col("number_of_text_reviews") /
    (F.col("number_of_text_reviews") + F.lit(confidence_saturation_parameter))
)

listing_features = listing_features.withColumn(
    "sentiment_multiplier",
    F.lit(1.0) +
    F.lit(max_sentiment_impact) *
    F.col("confidence_weight") *
    F.col("average_sentiment_score")
)

# clamp [0.9, 1.1]
listing_features = listing_features.withColumn(
    "sentiment_multiplier",
    F.when(F.col("sentiment_multiplier") < 0.8, F.lit(0.8))
     .when(F.col("sentiment_multiplier") > 1.2, F.lit(1.2))
     .otherwise(F.col("sentiment_multiplier"))
)

sentiment_multiplier_output = listing_features.select(
    "listing_id",
    "number_of_text_reviews",
    "number_of_all_reviews",
    "average_sentiment_score",
    "confidence_weight",
    "sentiment_multiplier"
)

sentiment_multiplier_output.show(truncate=False)




# COMMAND ----------

# save df to delta
sentiment_multiplier_output.write.mode("overwrite").saveAsTable("airbnb_sentiment_multiplier_df")

# COMMAND ----------

sentiment_multiplier_output.select("sentiment_multiplier").toPandas().plot.hist(bins=50)

# COMMAND ----------

min_max_sentiment = sentiment_multiplier_output.agg(
    F.min("sentiment_multiplier").alias("min_sentiment_multiplier"),
    F.max("sentiment_multiplier").alias("max_sentiment_multiplier")
)
display(min_max_sentiment)

# COMMAND ----------

# MAGIC %md
# MAGIC Add the relevant cols to the us_filtered_vfm df 

# COMMAND ----------

us_filtered_vfm = (
    us_filtered_vfm
    .join(
        sentiment_multiplier_output.select(
            F.col("listing_id").alias("property_id"),
            "sentiment_multiplier"
        ),
        on="property_id",
        how="left"
    )
    .withColumn(
        "sentiment_boosted_rating",
        F.col("ratings") * F.col("sentiment_multiplier")
    )
)

display(us_filtered_vfm)
us_filtered_vfm.select("sentiment_boosted_rating").toPandas().plot.hist(bins=50)

# COMMAND ----------

us_filtered_vfm.select("ratings").toPandas().astype({'ratings': 'float'}).plot.hist(bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create the VFM columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Relative price

# COMMAND ----------

from pyspark.sql import functions as F

# ================================
# Relative price vs city median
# Adds: city_median_price, relativePrice
# ================================
df = us_filtered_vfm
CITY_COL = "city"      
PRICE_COL = "price"    

# 1) compute median price per city (Spark percentile_approx)
city_medians = (
    us_filtered_vfm
    .filter(F.col(CITY_COL).isNotNull() & F.col(PRICE_COL).isNotNull())
    .groupBy(CITY_COL)
    .agg(F.expr(f"percentile_approx({PRICE_COL}, 0.5, 10000)").alias("city_median_price"))
)

# 2) join back + compute relativePrice (ratio to city median)
us_filtered_vfm = (
    us_filtered_vfm
    .join(city_medians, on=CITY_COL, how="left")
    .withColumn(
        "relativePrice",
        F.when(F.col("city_median_price").isNull() | (F.col("city_median_price") == 0), F.lit(None).cast("double"))
         .otherwise((F.col(PRICE_COL) / F.col("city_median_price")).cast("double"))
    )
)

display(us_filtered_vfm)


# COMMAND ----------

# MAGIC %md
# MAGIC ### WeightedRating

# COMMAND ----------

from pyspark.sql import functions as F

# ---- set your column names here ----
BOOST_COL = "sentiment_boosted_rating"               # R_b
V_COL     = "property_number_of_reviews_capped"      # v (total reviews)
M = 22                                               # m (smoothing) - the median of property_number_of_reviews_capped


# 1) global mean C of boosted rating
C = (us_filtered_vfm
     .select(F.avg(F.col(BOOST_COL)).alias("C"))
     .first()["C"])

# 2) WeightedRating
us_filtered_vfm = (us_filtered_vfm
      .withColumn(V_COL, F.coalesce(F.col(V_COL), F.lit(0)))
      .withColumn(BOOST_COL, F.coalesce(F.col(BOOST_COL), F.lit(C)))
      .withColumn(
          "WeightedRating",
          (F.col(V_COL) / (F.col(V_COL) + F.lit(M))) * F.col(BOOST_COL) +
          (F.lit(M) / (F.col(V_COL) + F.lit(M))) * F.lit(C)
      )
)

# df now has: WeightedRating
display(us_filtered_vfm)


# COMMAND ----------

# MAGIC %md
# MAGIC ### AmenityCount and Score

# COMMAND ----------

from pyspark.sql import functions as F, types as T

# ----------------------------
# CONFIG
# ----------------------------
AMENITIES_COL = "amenities"   # your column
CITY_COL = "city"             # set None for global thresholds
SCORE_COL_OUT = "amenityScore"

BASE_SAT_PCTL = 0.90   # p90
BONUS_PCTL    = 0.97   # p97
BONUS_WEIGHT  = 0.20   # up to +0.20

# ----------------------------
# 1) Parse amenities JSON (array<struct<group_name:string, items:array<struct<name:string,value:string>>>)
# ----------------------------
amen_schema = T.ArrayType(
    T.StructType([
        T.StructField("group_name", T.StringType(), True),
        T.StructField("items", T.ArrayType(
            T.StructType([
                T.StructField("name",  T.StringType(), True),
                T.StructField("value", T.StringType(), True),
            ])
        ), True),
    ])
)

amen_struct = F.from_json(F.col(AMENITIES_COL).cast("string"), amen_schema)

# Keep groups except "Not included", then flatten items, then take item.name
us_filtered_vfm = us_filtered_vfm.withColumn(
    "_amen_groups",
    F.when(amen_struct.isNull(), F.array().cast(amen_schema)).otherwise(amen_struct)
)

us_filtered_vfm = us_filtered_vfm.withColumn(
    "_amen_groups",
    F.expr("""
      filter(_amen_groups, g -> lower(trim(g.group_name)) <> 'not included')
    """)
)

us_filtered_vfm = us_filtered_vfm.withColumn(
    "_amen_items",
    F.when(F.col("_amen_groups").isNull(), F.array())
     .otherwise(F.flatten(F.expr("transform(_amen_groups, g -> g.items)")))
)

us_filtered_vfm = us_filtered_vfm.withColumn(
    "_amenities_arr",
    F.array_distinct(
        F.filter(
            F.transform(F.col("_amen_items"), lambda it: F.trim(it["name"])),
            lambda x: (x.isNotNull()) & (x != F.lit(""))
        )
    )
)

# ----------------------------
# 2) Count amenities
# ----------------------------
us_filtered_vfm = us_filtered_vfm.withColumn("amenitiesCount", F.size(F.col("_amenities_arr")))

# ----------------------------
# 3) Thresholds p90/p97 (global or per-city)
# ----------------------------
if CITY_COL is None:
    p90, p97 = us_filtered_vfm.approxQuantile("amenitiesCount", [BASE_SAT_PCTL, BONUS_PCTL], 0.01)
    p90 = max(float(p90), 1.0)
    p97 = max(float(p97), p90 + 1.0)
    us_filtered_vfm = us_filtered_vfm.withColumn("_p90", F.lit(p90)).withColumn("_p97", F.lit(p97))
else:
    thresh = (
        us_filtered_vfm.groupBy(F.col(CITY_COL).alias("_city_key"))
          .agg(
              F.expr(f"percentile_approx(amenitiesCount, {BASE_SAT_PCTL}, 10000)").alias("_p90"),
              F.expr(f"percentile_approx(amenitiesCount, {BONUS_PCTL}, 10000)").alias("_p97"),
          )
          .withColumn("_p90", F.greatest(F.col("_p90").cast("double"), F.lit(1.0)))
          .withColumn("_p97", F.greatest(F.col("_p97").cast("double"), F.col("_p90") + F.lit(1.0)))
    )

    us_filtered_vfm = us_filtered_vfm.join(thresh, us_filtered_vfm[CITY_COL] == thresh["_city_key"], "left").drop("_city_key")

# ----------------------------
# 4) Score: base saturation @p90 + bonus above p90 up to p97
# ----------------------------
us_filtered_vfm = us_filtered_vfm.withColumn("_base", F.least(F.col("amenitiesCount") / F.col("_p90"), F.lit(1.0)))

us_filtered_vfm = us_filtered_vfm.withColumn(
    "_bonus",
    F.when(F.col("amenitiesCount") <= F.col("_p90"), F.lit(0.0))
     .otherwise(
         F.least(
             (F.col("amenitiesCount") - F.col("_p90")) / (F.col("_p97") - F.col("_p90")),
             F.lit(1.0)
         ) * F.lit(BONUS_WEIGHT)
     )
)

us_filtered_vfm = us_filtered_vfm.withColumn(SCORE_COL_OUT, (F.col("_base") + F.col("_bonus")).cast("double"))

# ----------------------------
# 5) Cleanup
# ----------------------------
us_filtered_vfm = us_filtered_vfm.drop("_amen_groups", "_amen_items", "_amenities_arr", "_p90", "_p97", "_base", "_bonus")

us_filtered_vfm.select(CITY_COL if CITY_COL else F.lit("GLOBAL").alias("scope"),
          "amenitiesCount", SCORE_COL_OUT).show(10, truncate=False)


display(us_filtered_vfm)

# COMMAND ----------

percentiles = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
pvals_price = us_filtered_vfm.approxQuantile("price", percentiles, 0.01)

for p, v in zip(percentiles, pvals_price):
    print(f"price - {int(p*100)}th percentile: {v}")

# COMMAND ----------

percentiles = [0.045, 0.0475, 0.05]
pvals_price = us_filtered_vfm.approxQuantile("price", percentiles, 0.01)

for p, v in zip(percentiles, pvals_price):
    print(f"price - {int(p*100)}th percentile: {v}")

# COMMAND ----------

# Get 4th and 5th percentiles for price
p4, p5 = us_filtered_vfm.approxQuantile("price", [0.04, 0.05], 0.01)

# Filter for price between 4th and 5th percentile
price_4_5_df = us_filtered_vfm.filter((F.col("price") >= p4) & (F.col("price") <= p5))

# Plot histogram for price in this range
price_4_5_df.select("price").toPandas().plot.hist(bins=50, title="Price Histogram (4th-5th Percentile)")

# COMMAND ----------

# Get 4th percentile for price
p4 = us_filtered_vfm.approxQuantile("price", [0.04], 0.01)[0]

# Filter out rows where price is below the 4th percentile
us_filtered_vfm = us_filtered_vfm.filter(F.col("price") >= p4)

# Print the shape (rows, columns)
n_rows = us_filtered_vfm.count()
n_cols = len(us_filtered_vfm.columns)
print(f"Shape: ({n_rows}, {n_cols})")

# COMMAND ----------

us_filtered_vfm = us_filtered_vfm.dropDuplicates(["property_id"])
# Print the shape (rows, columns)
n_rows = us_filtered_vfm.count()
n_cols = len(us_filtered_vfm.columns)
print(f"Shape: ({n_rows}, {n_cols})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot hists of the signals

# COMMAND ----------

display(us_filtered_vfm.select("relativePrice", "WeightedRating", "amenityScore").summary())

# COMMAND ----------

us_filtered_vfm.select("relativePrice").toPandas().plot.hist(bins=50, title="relativePrice Histogram")
us_filtered_vfm.select("WeightedRating").toPandas().plot.hist(bins=50, title="WeightedRating Histogram")
us_filtered_vfm.select("amenitiesCount").toPandas().plot.hist(bins=50, title="amenitiesCount Histogram")
us_filtered_vfm.select("amenityScore").toPandas().plot.hist(bins=50, title="amenityScore Histogram")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardize the signals we made

# COMMAND ----------

df = us_filtered_vfm  

CITY_COL = "city"           
COLS = ["relativePrice", "WeightedRating", "amenityScore"]  # signals to standardize

# 1) per-city mean/std for each signal
agg_exprs = []
for c in COLS:
    agg_exprs += [
        F.avg(F.col(c)).alias(f"_{c}_mu"),
        F.stddev_pop(F.col(c)).alias(f"_{c}_sigma"),
    ]

city_stats = (
    df.groupBy(CITY_COL)
      .agg(*agg_exprs)
)

# 2) join and compute z-scores (safe if sigma=0/null)
df2 = df.join(city_stats, on=CITY_COL, how="left")

for c in COLS:
    sigma = F.when(
        F.col(f"_{c}_sigma").isNull() | (F.col(f"_{c}_sigma") == 0),
        F.lit(1.0)
    ).otherwise(F.col(f"_{c}_sigma"))

    df2 = df2.withColumn(f"{c}_z", (F.col(c) - F.col(f"_{c}_mu")) / sigma)

# 3) cleanup
drop_cols = []
for c in COLS:
    drop_cols += [f"_{c}_mu", f"_{c}_sigma"]

df2 = df2.drop(*drop_cols)

# overwrite / continue with df2
us_filtered_vfm = df2

display(us_filtered_vfm.select(CITY_COL, *COLS, *(f"{c}_z" for c in COLS)).limit(10))


# COMMAND ----------

us_filtered_vfm.select("relativePrice_z").toPandas().plot.hist(bins=50, title="Z score (Normalized) relativePrice Histogram")
us_filtered_vfm.select("WeightedRating_z").toPandas().plot.hist(bins=50, title="Z score (Normalized) WeightedRating Histogram")
us_filtered_vfm.select("amenityScore_z").toPandas().plot.hist(bins=50, title="Z score (Normalized) amenityScore Histogram")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalizing the scores to be [0,1]

# COMMAND ----------

from pyspark.sql import functions as F

RP_Z = "relativePrice_z"
WR_Z = "WeightedRating_z"
AS_Z = "amenityScore_z"

# logistic approx to normal CDF: Phi(z) ≈ 1 / (1 + exp(-1.702*z))
def norm_cdf_logistic(z_col: str):
    return (F.lit(1.0) / (F.lit(1.0) + F.exp(F.lit(-1.702) * F.col(z_col))))

us_filtered_vfm = (
    us_filtered_vfm
    .withColumn("relativePrice_01", (F.lit(1.0) - norm_cdf_logistic(RP_Z)).cast("double"))
    .withColumn("WeightedRating_01", (norm_cdf_logistic(WR_Z)).cast("double"))
    .withColumn("amenityScore_01", (norm_cdf_logistic(AS_Z)).cast("double"))
)

display(us_filtered_vfm.select("relativePrice_01","WeightedRating_01","amenityScore_01").summary())


# COMMAND ----------

us_filtered_vfm.select("relativePrice_01").toPandas().plot.hist(bins=50, title="[0,1] ]Normalized] relativePrice Histogram")
us_filtered_vfm.select("WeightedRating_01").toPandas().plot.hist(bins=50, title="[0,1] ]Normalized] WeightedRating Histogram")
us_filtered_vfm.select("amenityScore_01").toPandas().plot.hist(bins=50, title="[0,1] ]Normalized amenityScore Histogram")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add the baseline VFM

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

RP, WR, AS = "relativePrice_01", "WeightedRating_01", "amenityScore_01"
CITY = "city"   # change if needed
BETA = 0.3
EPS  = 1e-6     # prevents log(0) and division blowups

# log baseline (stable): log(WR/RP * (1+beta*AS))
df = us_filtered_vfm.withColumn(
    "baseline_vfm_log",
    F.log(F.greatest(F.coalesce(F.col(WR), F.lit(0.0)), F.lit(EPS))) -
    F.log(F.greatest(F.coalesce(F.col(RP), F.lit(0.0)), F.lit(EPS))) +
    F.log1p(F.lit(BETA) * F.coalesce(F.col(AS), F.lit(0.0)))
)

# optional: [0,1] within city (percentile)
w = Window.partitionBy(CITY).orderBy(F.col("baseline_vfm_log"))
df = df.withColumn("baseline_vfm_01", F.percent_rank().over(w).cast("double"))

us_filtered_vfm = df


# quick sanity
us_filtered_vfm.select("baseline_vfm_log", RP, WR, AS).summary("count","min","50%","max").show()

us_filtered_vfm.select("baseline_vfm_log").toPandas().plot.hist(bins=50, title="[0,1] Normalized Baseline VFM Histogram")



# COMMAND ----------


# quick sanity
us_filtered_vfm.select("baseline_vfm_log", RP, WR, AS).summary("count","min","50%","max").show()

us_filtered_vfm.select("baseline_vfm_log").toPandas().plot.hist(bins=50, title="[0,1] Normalized log(Baseline VFM) Histogram")



# COMMAND ----------

# MAGIC %md
# MAGIC  **Maybe shift to positive scale [0,27]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Saving the df

# COMMAND ----------


us_filtered_vfm.write.mode("overwrite").format("delta").saveAsTable("us_filtered_vfm")


# COMMAND ----------

listings = spark.read.table("us_filtered_vfm")

# COMMAND ----------

# MAGIC %md
# MAGIC Loading worked

# COMMAND ----------

# MAGIC %md
# MAGIC ---------------