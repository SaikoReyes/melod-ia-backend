class Config:
    #JWT CONFIG
    SECRET_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6OSwiZXhwIjoxNzEzODE3MzQ3fQ.GmHu8yBhJvmqNjQrHETBblpJX8lwAN7KXFOP0sg4XzA"

    #DB CONFIG
    MYSQL_HOST = 'melodia-server.mysql.database.azure.com'
    #MYSQL_HOST = 'localhost'
    MYSQL_USER = 'melodia'
    #MYSQL_USER = 'root'
    MYSQL_PASSWORD = 'Password123!'
    #MYSQL_PASSWORD = 'password'
    MYSQL_DB = 'melodiadb'

    #DEBUG
    DEBUG = True
