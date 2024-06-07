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
# 1. Вывод фильмов с участием Тома Круза
tom_cruise_movies = df.filter(
    (col("Stars").contains("TomCruise")) &
    (col("Duration") > 100)
)

# Сортировка фильмов сначала по убыванию, затем по возрастанию рейтинга IMDb
best_to_worst_movies = tom_cruise_movies.orderBy(desc("IMDb-Rating"))
worst_to_best_movies = tom_cruise_movies.orderBy(asc("IMDb-Rating"))

# Перевод длительности в часы
best_to_worst_movies = best_to_worst_movies.withColumn("Duration (Hours)", col("Duration") / 60)
worst_to_best_movies = worst_to_best_movies.withColumn("Duration (Hours)", col("Duration") / 60)


# 2. Витрина лучших комедий до 2000 года
best_comedies_pre_2000 = df.filter(
    (col("Category").contains("Comedy")) &
    (col("ReleaseYear") < 2000)
).orderBy(desc("IMDb-Rating"))

# Показ результатов
print("Best Comedies Before 2000:")
best_comedies_pre_2000.show()

# 3. Витрина худших фильмов в жанре драма
worst_dramas = df.filter(
    col("Category").contains("Drama")
).orderBy(asc("IMDb-Rating"))

# Показ результатов
print("Worst Dramas:")
worst_dramas.show()

# 4. График самых топовых жанров за последние 50 лет
# Фильтрация фильмов за последние 50 лет
df_filtered = df.filter(col("ReleaseYear") >= 1973)

# Группировка по жанру и вычисление среднего рейтинга
top_genres = df_filtered.groupBy("Category").agg(avg("IMDb-Rating").alias("Average_Rating"))

# Фильтрация значений None в столбце "Category"
top_genres = top_genres.filter(col("Category").isNotNull())

# Конвертация в pandas DataFrame для построения графика
top_genres_pd = top_genres.toPandas()

# Построение графика
plt.figure(figsize=(10, 6))
plt.barh(top_genres_pd["Category"], top_genres_pd["Average_Rating"], color='skyblue')
plt.xlabel('Average IMDb Rating')
plt.title('Top Genres in the Last 50 Years')
plt.gca().invert_yaxis()
plt.show()


# 5. Топ 5 режиссеров за последние 20 лет
# Фильтрация фильмов за последние 20 лет
df_last_20_years = df.filter(col("ReleaseYear") >= 2003)

# Группировка по режиссеру и вычисление среднего рейтинга и времени
top_directors = df_last_20_years.groupBy("Director").agg(
    mean("IMDb-Rating").alias("Average_Rating"),
    mean("Duration").alias("Average_Duration"),
    count("Title").alias("Movie_Count")
).orderBy(desc("Average_Rating"))

# Фильтрация топ 5 режиссеров
top_5_directors = top_directors.limit(5)

# Показ результатов
print("Top 5 Directors in the Last 20 Years:")
top_5_directors.show()
