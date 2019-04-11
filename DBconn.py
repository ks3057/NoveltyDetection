__author__ = "KirtanaSuresh"

import mysql.connector
from mysql.connector import errorcode
import csv

try:
    conn = mysql.connector.connect(
        unix_socket='/Applications/MAMP/tmp/mysql/mysql.sock',
        user="root",
        password="root",
        host="localhost",
        database="crowdRE",
        raise_on_warnings=True,
    )
    # print("It works")
except mysql.connector.Error as e:
    if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("wrong username and password")
    else:
        print(e)

cursor = conn.cursor()

cursor.execute("select r.id, r.user_id, r.role, r.feature, r.benefit, "
               "r.application_domain, AVG("
               "rr.novelty)"
               " from requirements r join "
               "requirements_ratings rr "
               "on r.id = rr.requirement_id"
               " group by r.id, r.user_id, r.role, r.feature, r.benefit, "
               "r.application_domain"
               );

rows = cursor.fetchall()
fp = open('novelty_avg.csv', 'w')
myFile = csv.writer(fp)
myFile.writerows(rows)
fp.close()

