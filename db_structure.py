from peewee import *

db = SqliteDatabase('results.sqlite3')

class Result(Model):
    distribution = IntegerField()
    db_size = IntegerField()
    betat = DoubleField()
    nhlayers = IntegerField()
    hl_nnodes = IntegerField()
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db

Result.create_table()
