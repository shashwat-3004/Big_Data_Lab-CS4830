#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml import Pipeline
#from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
#from sparknlp.base import DocumentAssembler, Finisher
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF,HashingTF,Tokenizer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[ ]:


spark.stop()


# In[2]:


spark=SparkSession.builder.appName("yelp").config('spark.driver.memory','2g').config('saprk.executor.memory','4g').getOrCreate()


# In[ ]:


spark.stop()


# In[3]:


data_schema = StructType([StructField('id_1', StringType(), True),
               StructField('cfu_1', FloatType(), True),
               StructField('date', StringType(), True),
               StructField('cfu_2', FloatType(), True),
               StructField('id_2', StringType(), True),
               StructField('stars', FloatType(), True),
               StructField('text', StringType(), True),
               StructField('cfu_3', FloatType(), True),
               StructField('id_3', StringType(), True)])


# In[4]:


yelp_dataset = spark.read.format("csv").schema(data_schema).option("mode", "DROPMALFORMED").option("quote", '"').option("multiline", "true").option("escape", "\"").load(["gs://shapap2/YELP_train.csv/part-00000-808f9971-b2b6-4a6f-b8cf-0822a68f365f-c000.csv",
     "gs://shapap2/YELP_train.csv/part-00001-808f9971-b2b6-4a6f-b8cf-0822a68f365f-c000.csv"])


# In[5]:


yelp_dataset=yelp_dataset.filter((yelp_dataset.stars>=0)&(yelp_dataset.stars<=5))
yelp_dataset = yelp_dataset.filter("stars is NOT NULL AND text is NOT NULL") 
print('the final Yelp dataset has ' + str(yelp_dataset.count()) + ' rows.')


# In[6]:


training_data, test_data = yelp_dataset.randomSplit([0.8, 0.2], seed = 42)


# In[7]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopword_remover = StopWordsRemover(inputCol = "words", outputCol = "filtered")
tf = HashingTF(inputCol = "filtered", outputCol = "raw_features")
idf = IDF(inputCol = "raw_features", outputCol = "features")


# In[8]:



dt=DecisionTreeClassifier(featuresCol = 'features', labelCol = 'stars',maxDepth=10)
pipe = Pipeline(
    stages = [tokenizer, stopword_remover, tf, idf, dt]
)


# In[ ]:



print("Fitting the model")
lr_model = pipe.fit(training_data)
print('Training done.')


# In[ ]:


dt_evaluator_f1 = MulticlassClassificationEvaluator(predictionCol = "prediction", 
                                  labelCol = "stars", metricName = "f1")

dt_evaluator_acc=MulticlassClassificationEvaluator(predictionCol = "prediction", 
                                  labelCol = "stars", metricName = "accuracy")

# training prediction
prediction_train = lr_model.transform(training_data)
print('Transforming training data done.')
prediction_train.select("prediction", "stars", "features").show(5)
print("Evaluating on Training data(F1-score)):", dt_evaluator_f1.evaluate(prediction_train))
print("Evaluating on Training data(Accuracy)):", dt_evaluator_acc.evaluate(prediction_train))


# In[ ]:


prediction_test = lr_model.transform(test_data)
print("F1:", dt_evaluator_f1.evaluate(prediction_test))
print("Accuracy:",dt_evaluator_acc.evaluate(prediction_test))


# In[ ]:




