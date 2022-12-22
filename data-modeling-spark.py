from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession \
.builder \
.appName("") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()
data=spark.read.csv("gs://yg2537_6893_data/data_clean.csv",inferSchema=False, header='true')
data=data.drop("_c0")
data=data.withColumn("price",data.price.cast('double'))
data=data.withColumn("bedrooms",data.bedrooms.cast('double'))
data=data.withColumn("bathrooms",data.bathrooms.cast('double'))
data=data.withColumn("floors",data.floors.cast('double'))
data=data.withColumn("condition",data.condition.cast('int'))

stages = [];
categoricalColumns=['sqft_living','sqft_lot','waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated']
for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol=categoricalCol,outputCol=categoricalCol + "Index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol="price", outputCol="label")
stages += [label_stringIdx]

numericCols = ["bedrooms", "bathrooms", "floors", "condition"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(data)
preppedDataDF = pipelineModel.transform(data)
cols = data.columns
selectedcols = ["label", "features"] + cols
data = preppedDataDF.select(selectedcols)




trainingData,testData=preppedDataDF.randomSplit([0.7,0.3],seed=100)
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainingData)
trainingSummary = lrModel.summary
print("r2: %f" % trainingSummary.r2)
print("Coefficients: %s" % str(lrModel.coefficients))



stages = [];
categoricalColumns=['sqft_living','sqft_lot','waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated']
for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol=categoricalCol,outputCol=categoricalCol + "Index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol="price", outputCol="label")
stages += [label_stringIdx]

numericCols = ["bedrooms", "bathrooms", "floors", "condition"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

featureIndexer =VectorIndexer(inputCol='features', outputCol="indexedFeatures", maxCategories=4).fit(data)


gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
pipeline = Pipeline(stages=[featureIndexer, gbt])
model = pipeline.fit(trainingData)


gbtModel = model.stages[1]
print(gbtModel.featureImportances)