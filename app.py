import jwt
import datetime
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL, MySQLdb
from flask_bcrypt import Bcrypt
from flask import Flask
from flask_cors import CORS
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# Configuración de la base de datos
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'melodiadb'

mysql = MySQL(app)

SECRET_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6OSwiZXhwIjoxNzEzODE3MzQ3fQ.GmHu8yBhJvmqNjQrHETBblpJX8lwAN7KXFOP0sg4XzA"

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            if token.startswith('Bearer '):
                token = token[7:]
            payload = jwt.decode(token,SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        except TypeError:
            return jsonify({'message': 'An error occurred during token decoding'}), 500

        return f(*args, **kwargs)

    return decorated_function


@app.route('/refresh_token', methods=['POST'])
def refresh_token():
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        return jsonify({'message': 'Token is missing or invalid'}), 401
    
    try:
        decoded_token = jwt.decode(token[7:], SECRET_KEY, algorithms=['HS256'], options={"verify_exp": False})
        exp = datetime.utcfromtimestamp(decoded_token['exp'])
        now = datetime.utcnow()
        if exp - now < timedelta(seconds=60):
            new_token = jwt.encode({
                'id': decoded_token['id'],
                'exp': datetime.utcnow() + timedelta(minutes=5)
            }, SECRET_KEY, algorithm='HS256')
            return jsonify({'token': new_token}), 200
        else:
            return jsonify({'message': 'Token not yet ready for renewal'}), 400
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token provided'}), 401


@app.route('/registro', methods=['POST'])
def register():
    email = request.json['email']
    plain_text_password = request.json['password']
    nombre = request.json['nombre']
    apellido = request.json['apellido']
    fechaNacimiento = request.json['fechaNacimiento']
    hashed_password = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')

    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO usuarios (email, contraseña, nombre, apellido,fechanac) VALUES (%s, %s, %s, %s, %s)",
            (email, hashed_password, nombre, apellido,fechaNacimiento)
        )
        mysql.connection.commit()
        response = {'message': 'Usuario registrado exitosamente'}
        status_code = 201
    except Exception as e:
        mysql.connection.rollback()
        response = {'error': str(e)}
        status_code = 400
    finally:
        cursor.close()

    return jsonify(response), status_code

@app.route('/login', methods=['POST'])
def login():
    email = request.json['email']
    password_candidate = request.json['password']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        cursor.execute("SELECT * FROM usuarios WHERE email = %s", [email])
        user = cursor.fetchone()
        if user and bcrypt.check_password_hash(user['contraseña'], password_candidate):
            token = jwt.encode({
                'id': user['idUsuario'],
                'exp': datetime.utcnow() + timedelta(minutes=5)
            }, SECRET_KEY, algorithm='HS256')
            user_data = {
                'idUsuario': user['idUsuario'],
                'email': user['email'],
                'nombre': user['nombre'],
                'apellido': user['apellido'],
                'token': token
            }
            return jsonify(user_data), 200
        else:
            return jsonify({'error': 'Usuario o contraseña incorrectos'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()

@app.route('/usuarios')
def usuarios():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM usuarios")
    resultados = cur.fetchall()
    cur.close()
    return jsonify(resultados)


if __name__ == '__main__':
    
    app.run(port=8000, debug=True)
