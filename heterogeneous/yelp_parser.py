"""
Parse yelp files and create edge lists for node2vec
"""
import os
import csv

# configure spark to be started with more allocated memory
memory = '14g'
pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import BooleanType, ArrayType, StringType, StructType, StructField, IntegerType
from pyspark.sql import Row
from pyspark.sql.functions import udf, array_contains, col, size, explode

from src import node2vec

# seting up spark
os.environ["PYSPARK_PYTHON"] = "python3.5"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3.5"

conf = (SparkConf()
        .setMaster("local")
        .setAppName("Yelp-parser")
        .set("spark.executor.cores", "8")
        .set("spark.executor.memory", "1g")
        )

sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

# global variables
YELP_PATH = "/home/natalia/Projects/network-analysis/Yelp/yelp_dataset_challenge_round9"
KEYWORD_NUM = 100
BUSINESS_LIM = None


def read_yelp():
    """
    Read yelp json files and convert to networkx graph
    :return:
    """
    # paths to json files
    business_path = os.path.join(YELP_PATH, "yelp_academic_dataset_business.json")
    checkin_path = os.path.join(YELP_PATH, "yelp_academic_dataset_checkin.json")
    review_path = os.path.join(YELP_PATH, "yelp_academic_dataset_review.json")
    tip_path = os.path.join(YELP_PATH, "yelp_academic_dataset_tip.json")
    user_path = os.path.join(YELP_PATH, "yelp_academic_dataset_user.json")

    business_ids = read_yelp_business(business_path, limit=BUSINESS_LIM)
    user_review, review_business, review_keyword,\
    review_dict, keyword_dict, user_dict, business_dict = read_yelp_review(review_path, business_ids)

    return user_review, review_business, review_keyword, review_dict, keyword_dict, user_dict, business_dict


def read_yelp_business(file_path, limit=None):
    """
    Method to read business json file
    :param limit: integer
    :return: return list of business ids
    """
    print("Reading yelp business json file...")
    df = sql_context.read.json(file_path)
    df.printSchema()
    # filter those businesses which contain "fast food, American, sushi bar"
    filter_categories = ["fast food", "american", "sushi bar"]

    def exists(f):
        return udf(lambda xs: any(f(x) for x in xs) if xs else False, BooleanType())

    # business_ids = df.where(exists(lambda x: x.lower() in filter_categories)("categories"))\
    #     .select("business_id").distinct()\
    #     .rdd.flatMap(lambda x: x).collect()
    # filter just sushi stuff!
    business_ids = df.where(exists(lambda x: "sushi" in x.lower())("categories")) \
        .select("business_id").distinct() \
        .rdd.flatMap(lambda x: x)
    if limit:
        business_ids = business_ids.take(limit)
    else:
        business_ids = business_ids.collect()
    print("Number of businesses to be used: {}".format(len(business_ids)))
    print("Sample of business ids: {}".format(business_ids[:5]))
    return business_ids


def tokenize_reviews(review_df):
    """
    Tokenize review text and do some filtering on tokens.
    :param review_df:
    :return:
    """
    print("Tokenizing review texts...")
    # tokenize texts of reviews
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenized = tokenizer.transform(review_df)
    print("Tokenized: ", tokenized.show(5))
    # remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="tokens")
    tokens = remover.transform(tokenized)
    print("Extracted tokens: ", tokens.show(5))

    # TODO: This can be further improved by stemming words, removing jargon, filtering English etc.
    # filter tokens which contain only ascii alpha, are longer than 3 characters
    is_ascii = lambda s: len(s) == len(s.encode())
    def alpha_filter(items):
        return [i for i in items if i.isalpha() and len(i) > 3 and is_ascii(i)]

    letters = udf(lambda items: alpha_filter(items), ArrayType(StringType()))
    tokens = tokens.select("review_id", "tokens",
                           letters(col("tokens")).alias("filteredTokens"))
    print("Filtered tokens")
    tokens.show(5)

    return tokens


def get_tfidf_reviews(tokens):
    """
    Compute tf-idf for reviews when they are tokenized.
    :param tokens: Spark dataframe which contains terms from tokenized review text
    :return: list of max tf-idf per term, list of terms
    """
    print("Obtaining tfidf for review texts...")
    # getting counts: CountVectorizer is used instead of HashingTF so that I can lookup words
    cv = CountVectorizer(inputCol="filteredTokens", outputCol="rawFeatures",
                         vocabSize=2*KEYWORD_NUM, minDF=5).fit(tokens)
    # obtaining tf vector
    tf_df = cv.transform(tokens)
    print("tf df: ", tf_df.show(5))

    # idf
    idf_model = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5).fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)
    print("tfidf df: ", tfidf_df.select("filteredTokens", "features").show(5))

    # getting tf-idf per term
    print("Reducing...")

    # TODO: we can change max to average?
    def tfidf_reduce(a, b):
        # here we select max TF-IDF for each token in the whole corpora
        return [max(ab) for ab in zip(a, b)]

    features = tfidf_df.select("features").rdd \
        .map(lambda x: x[0].toArray()) \
        .reduce(lambda x1, x2: tfidf_reduce(x1, x2))

    print("features: ", features)
    print("vocab: ", cv.vocabulary)

    return features, cv.vocabulary


def extract_keywords(review_df, num=100):
    """
    Extract keywords from review text, rank according to TF-IDF and select top num.
    :param review_df:
    :param num: integer to select top ranked keywords
    :return:
    """
    tokens = tokenize_reviews(review_df)

    features, vocabulary = get_tfidf_reviews(tokens)

    print("len vocab: ", len(vocabulary))
    print("len features: ", len(features))
    print(type(features))

    sorted_words = sorted(zip(vocabulary, features), key=lambda x: x[1], reverse=True)
    print("sorted words: ", list(sorted_words))

    # select top num words as keywords
    keywords = set(w[0] for w in sorted_words[:num])
    print("selected list of keywords: ", keywords)

    # identify list of keywords per review
    def keyword_filter(items):
        return list(set(items).intersection(keywords))

    filtered_keywords = udf(lambda items: keyword_filter(items), ArrayType(StringType()))
    review_keyword = tokens.select("review_id", filtered_keywords(col("filteredTokens")).alias("keywords"))\
        .where(size(col("keywords")) > 0).withColumn("keywords", explode("keywords"))
    print("Keywords and review: ")
    review_keyword.show(10)

    return review_keyword, keywords


def get_customers(review_df, review_dict, user_dict):
    """
    Construct a list of users and associated reviews.
    We change ids to the ones which will be used in the graph.
    :param review_df: spark dataframe
    :param review_dict: lookup dictionary for review ids
    :param user_dict: lookup dictionary for user ids
    :return:
    """
    user_convert = udf(lambda x: user_dict[x])
    review_convert = udf(lambda x: review_dict[x])

    return review_df.select(user_convert(col("user_id")).alias("from_node_id"),
                            review_convert(col("review_id")).alias("to_node_id"))


def get_business(review_df, review_dict, business_dict):
    """
    Construct a list of users and associated reviews.
    We change ids to the ones which will be used in the graph.
    :param review_df: spark dataframe
    :param review_dict: lookup dictionary for review ids
    :param user_dict: lookup dicionary for user ids
    :return:
    """
    business_convert = udf(lambda x: business_dict[x])
    review_convert = udf(lambda x: review_dict[x])

    return review_df.select(review_convert(col("review_id")).alias("from_node_id"),
                            business_convert(col("business_id")).alias("to_node_id"))


def get_keywords(review_df, review_dict, keyword_dict):
    """
    Construct a list of users and associated reviews.
    We change ids to the ones which will be used in the graph.
    :param review_df: spark dataframe
    :param review_dict: lookup dictionary for review ids
    :param user_dict: lookup dicionary for user ids
    :return:
    """
    keyword_convert = udf(lambda x: keyword_dict[x])
    review_convert = udf(lambda x: review_dict[x])

    return review_df.select(review_convert(col("review_id")).alias("from_node_id"),
                            keyword_convert(col("keywords")).alias("to_node_id"))


def create_node_ids(reviews, keywords, users, business):
    """
    Generate unique ids for all node types!
    :param reviews: list of review ids
    :param keywords: list of keywords
    :param users: list of user ids
    :param business: list of business ids
    :return:
    """
    review_dict = {rev: idx for idx, rev in enumerate(reviews)}
    node_counter = len(reviews) + 1

    keyword_dict = {w: idx + node_counter for idx, w in enumerate(keywords)}
    node_counter += len(keywords) + 1

    user_dict = {w: idx + node_counter for idx, w in enumerate(users)}
    node_counter += len(users) + 1

    business_dict = {w: idx + node_counter for idx, w in enumerate(business)}

    return review_dict, keyword_dict, user_dict, business_dict


def read_yelp_review(file_path, business_ids=None):
    """
    Method to read review json file, filter reviews for selected businesses and
    extract keywords.
    :param business_ids: list of business ids for which reviews will be read in
    :return:
    """
    print("Reading yelp review json file...")
    df = sql_context.read.json(file_path)
    df.printSchema()
    # filter reviews for selected businesses
    reviews = df.where(col("business_id").isin(business_ids)) \
        .select("review_id", "business_id", "user_id", "text")
    print("Number of selected reviews: {}".format(reviews.count()))

    review_keyword, keywords = extract_keywords(reviews, num=KEYWORD_NUM)  # review_id, keyword

    review_dict, keyword_dict, user_dict, business_dict = create_node_ids(
        reviews.select("review_id").distinct().rdd.flatMap(lambda x: x).collect(),
        keywords,
        reviews.select("user_id").distinct().rdd.flatMap(lambda x: x).collect(),
        business_ids)

    # user_node_id, review_node_id
    user_review = get_customers(reviews, review_dict=review_dict, user_dict=user_dict)
    # review_node_id, business_node_id
    review_business = get_business(reviews, review_dict=review_dict, business_dict=business_dict)
    # review_node_id, keyword_node_id
    review_keyword = get_keywords(review_keyword, review_dict=review_dict, keyword_dict=keyword_dict)

    return user_review, review_business, review_keyword, review_dict, keyword_dict, user_dict, business_dict

# TODO: read users
# TODO: read tips
# TODO: read checkins


def write_edge_lists(user_review, review_business, review_keyword, filepath):
    """
    Write edge list for heterogeneous network to the file.
    :param user_review:
    :param review_business:
    :param review_keyword:
    :param filepath:
    :return:
    """
    print("Concatenating all dataframes")
    df_concat = user_review.unionAll(review_business).unionAll(review_keyword)
    df_concat.show(10)
    print("Writing to the file {}".format(filepath))
    df_concat.write.option("sep", " ").option("header", "false").csv(filepath)
    return df_concat


def write_separate_edge_lists(user_review, review_business, review_keyword, dirpath):
    """
    Write edge lists for each type of edges to a separate file.
    :param user_review: Spark dataframe
    :param review_business: Spark dataframe
    :param review_keyword: Spark dataframe
    :param dirpath: path to the directory where edgelists will be written
    :return:
    """
    print("Writing user_review")
    filepath = os.path.join(dirpath, "yelp-sushi-user-review.edgelist")
    user_review.coalesce(1).write.option("sep", " ").option("header", "false").csv(filepath)
    print("Writing review_business")
    filepath = os.path.join(dirpath, "yelp-sushi-review-business.edgelist")
    review_business.coalesce(1).write.option("sep", " ").option("header", "false").csv(filepath)
    print("Writing review_keyword")
    filepath = os.path.join(dirpath, "yelp-sushi-review-keyword.edgelist")
    review_keyword.coalesce(1).write.option("sep", " ").option("header", "false").csv(filepath)
    return


def write_dictionaries(review_dict, keyword_dict, user_dict, business_dict, filepath):
    """
    Write lookup dictionaries for node ids
    :param review_dict: dictionary
    :param keyword_dict: dictionary
    :param user_dict: dictionary
    :param business_dict: dictionary
    :param filepath: string
    :return:
    """
    print("Writing dictionaries...")
    # node_id, type, content
    with open(filepath, "w+") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["node_id", "type", "original_id"])  # header
        rows = [[val, "review", key] for key, val in review_dict.items()]
        csvwriter.writerows(rows)
        rows = [[val, "business", key] for key, val in business_dict.items()]
        csvwriter.writerows(rows)
        rows = [[val, "keyword", key] for key, val in keyword_dict.items()]
        csvwriter.writerows(rows)
        rows = [[val, "user", key] for key, val in user_dict.items()]
        csvwriter.writerows(rows)


if __name__ == "__main__":
    user_review, review_business, review_keyword, \
    review_dict, keyword_dict, user_dict, business_dict = read_yelp()

    user_review.show(5)
    review_business.show(5)
    review_keyword.show(5)

    # all_df = write_edge_lists(user_review, review_business, review_keyword, "yelp_review_sushi.edgelist")

    dir_path = "/home/natalia/Projects/network-analysis/node2vec/graph/yelp"
    write_dictionaries(review_dict, keyword_dict,
                       user_dict, business_dict,
                       os.path.join(dir_path, "yelp_lookup_dictionaries.csv"))
    write_separate_edge_lists(user_review, review_business, review_keyword, dir_path)

