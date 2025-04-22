import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="",
        password="",
        database="face_recognition"
    )
