from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, length, split, desc, asc, avg, mean, count
from pyspark.sql.types import BooleanType, StringType, FloatType
import matplotlib.pyplot as plt

# Создание сессии Spark
spark = SparkSession.builder.appName("IMDbMovies").getOrCreate()

# Загрузка данных из CSV с помощью PySpark
df = spark.read.csv('IMDb_Data_final.csv', header=True, inferSchema=True)

# Удаление лишних пробелов в названиях столбцов
df = df.select([col(column).alias(column.strip()) for column in df.columns])

# Преобразование длительности фильма из "130min" в минуты
def parse_duration(duration):
    if duration is not None:
        return int(duration.replace('min', ''))
    return None

parse_duration_udf = udf(parse_duration, FloatType())
df = df.withColumn("Duration", parse_duration_udf(col("Duration")))

# Вывод первых нескольких строк для проверки формата данных
df.show(5)

# Проверка уникальных значений в столбце "Stars"
df.select("Stars").distinct().show(5)
