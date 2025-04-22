import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="admingriffith",
        password="Gr1ff1th@",
        database="face_recognition"
    )
