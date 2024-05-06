import pyphen

def separar_silabas_en_una_lista(texto):
    # Crea un diccionario de higienización para el idioma español
    dic = pyphen.Pyphen(lang='es')
    
    # Separa el texto en palabras
    palabras = texto.split()
    
    # Crea una lista para almacenar todas las sílabas
    todas_las_silabas = []
    
    # Recorre cada palabra, la divide en sílabas y agrega las sílabas a la lista
    for palabra in palabras:
        silabas = dic.inserted(palabra, hyphen='-').split('-')
        todas_las_silabas.extend(silabas)  # Usa extend para agregar elementos de la lista de sílabas a la lista principal
    
    return todas_las_silabas

# Ejemplo de uso
texto = "Hola mundo"
silabas = separar_silabas_en_una_lista(texto)
print(silabas)
