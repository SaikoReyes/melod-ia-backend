import jwt
import datetime
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL, MySQLdb
from flask_bcrypt import Bcrypt

app = Flask(__name__)
bcrypt = Bcrypt(app)

# Configuración de la base de datos
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'melodiadb'

mysql = MySQL(app)

SECRET_KEY = "tu_clave_secreta_para_jwt"

@app.route('/registro', methods=['POST'])

def register():
    # Obtener datos del request
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
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
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
