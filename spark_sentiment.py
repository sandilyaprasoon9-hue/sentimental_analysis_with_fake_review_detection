from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("BigDataSentiment").getOrCreate()

# Load data
df = spark.read.csv("data/Reviews.csv", header=True, inferSchema=True)

df = df.select("Text", "Score")
df = df.withColumnRenamed("Text", "review") \
       .withColumnRenamed("Score", "rating")

# Label
df = df.withColumn(
    "label",
    when(col("rating") >= 4, 1).otherwise(0)
)

# NLP Pipeline
tokenizer = Tokenizer(inputCol="review", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="features", numFeatures=5000)

lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, tf, lr])

# Train model
model = pipeline.fit(df)

# Predict
predictions = model.transform(df)

predictions.select("review", "prediction").show(5)
