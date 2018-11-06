from peewee import *
import os

pgdb = os.environ['pgdb']
pguser = os.environ['pguser']
pgpass = os.environ['pgpass']
pghost = os.environ['pghost']
pgport = os.environ['pgport']

db = PostgresqlDatabase(pgdb, user=pguser, password=pgpass,
host=pghost, port=pgport)
#db = SqliteDatabase('results.sqlite3')

class Result(Model):
    distribution = IntegerField()
    db_size = IntegerField()
    betat = DoubleField()
    complexity = IntegerField()
    estimator = TextField()
    method = TextField()
    retrain_permutations = IntegerField()
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db

Result.create_table()
