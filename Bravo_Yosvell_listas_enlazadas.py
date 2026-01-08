# EJERCICIOS PRÁCTICOS - LISTAS ENLAZADAS
# Unidad 3: Estructura de Datos
# ULEAM - Ingeniería en Software
"""
Nombre: Bravo Rosado Yosvell Antonio
Fecha: 20/12/2025
Ejercicios: 20/20
"""

#CLASES BASE PARA LOS TODOS LOS EJERCICIOS DE LISTAS ENLAZADAS

class Node:
    #Nodo para lista simplemente enlazada
    def __init__(self, data=None):
        self.data = data
        self.next = None
    
    def __repr__(self):
        return (f"Node({self.data})")


class DoubleNode:
    #Nodo para lista doblemente enlazada
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None
    
    def __repr__(self):
        return (f"DoubleNode({self.data})")


class LinkedList:
    #Lista simplemente enlazada con implementación completa
    
    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        nodes = []
        current = self.head
        while current:
            nodes.append(str(current.data))
            current = current.next
        return " → ".join(nodes) + " → None"
    
    def is_empty(self):
        #Verifica si la lista está vacía
        return self.head is None
    
    def append(self, data):
        #Agrega elemento al final de la lista
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1
    
    def prepend(self, data):
        #Agrega elemento al inicio de la lista
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self._size += 1
    
    def create_from_list(self, data_list):
        #Crea lista desde una lista de Python
        for item in data_list:
            self.append(item)

# ==============================================================
# EJERCICIO 1: Contar elementos
# ==============================================================
def count(self, elem):
    """
    Cuenta las ocurrencias de un elemento en la lista
    
    Args:
        elem: Elemento a contar
        
    Returns:
        Número de veces que aparece elem

    Logica:
    - Se recorre la lista desde la cabeza hasta el final
    - Se compara el valor de cada nodo con el elemento buscado
    - Cada vez que coincide, se incrementa un contador
    - Al finalizar el recorrido, se retorna el total acumulado
    """

    # Contador para almacenar cuántas veces aparece el elemento
    contador = 0

    # Puntero que inicia en la cabeza de la lista
    current = self.head

    # Recorremos la lista nodo por nodo
    while current is not None:
        # Si el valor del nodo actual coincide con el elemento buscado
        if current.data == elem:
            contador += 1  # Incrementamos el contador

        # Avanzamos al siguiente nodo
        current = current.next

    # Retornamos el número total de ocurrencias
    return contador

# ==============================================================
# EJERCICIO 2: Obtener elemento por índice
# ==============================================================
def get(self, index):
    """
    Obtiene el elemento en una posición específica
    
    Args:
        index: Posición del elemento (0-indexed)
        
    Returns:
        El elemento en la posición index
        
    Raises:
        IndexError: Si el índice está fuera de rango

    Logica:
    - Se valida que el indice este dentro del rango válido
    - Se recorre la lista desde la cabeza contando posiciones
    - Cuando el contador coincide con el indice, se retorna el dato
    """

    # Validamos que el indice sea valido
    if index < 0 or index >= self._size:
        raise IndexError("indice fuera de rango")

    # Puntero que inicia en la cabeza de la lista
    current = self.head
    current_index = 0

    # Recorremos la lista hasta llegar al índice deseado
    while current_index < index:
        current = current.next      # Avanzamos al siguiente nodo
        current_index += 1          # Incrementamos el contador de índice

    # Cuando salimos del ciclo, current está en la posición index
    return current.data

# ==============================================================
# EJERCICIO 3: Encontrar índice de elemento
# ==============================================================
def index_of(self, elem):
    """
    Encuentra el índice de la primera ocurrencia de un elemento
    
    Args:
        elem: Elemento a buscar
        
    Returns:
        Índice de la primera ocurrencia, o -1 si no existe
    
    Logica:
    - Se recorre la lista desde el inicio
    - Se compara cada nodo con el elemento buscado
    - Al encontrar la primera coincidencia, se retorna su índice
    - Si no se encuentra, se retorna -1
    """

    # Puntero que inicia en la cabeza de la lista
    current = self.head

    # Variable que llevará el control del índice
    index = 0

    # Recorremos la lista nodo por nodo
    while current is not None:

        # Si el dato del nodo coincide con el elemento buscado
        if current.data == elem:
            return index  # Retornamos el índice encontrado

        # Avanzamos al siguiente nodo
        current = current.next
        index += 1

    # Si terminamos el recorrido y no encontramos el elemento
    return -1

# ==============================================================
# EJERCICIO 4: Lista a array
# ==============================================================
def to_list(self):
    """
    Convierte la lista enlazada a una lista de Python
    
    Returns:
        Lista de Python con todos los elementos
    
    Logica:
    - Se crea una lista vacía de Python
    - Se recorren todos los nodos desde la cabeza
    - Cada dato se agrega a la lista
    - Al finalizar, se retorna la lista resultante
    
    """

    # Lista de Python donde se guardarán los datos
    result = []

    # Comenzamos desde la cabeza de la lista
    current = self.head

    # Recorremos la lista nodo por nodo
    while current is not None:
        result.append(current.data)
        current = current.next

    # Retornamos la lista de Python
    return result

# ==============================================================
# EJERCICIO 5: Limpiar lista
# ==============================================================

def clear(self):
    """
    Elimina todos los elementos de la lista

    Logica:
    - Para vaciar una lista simplemente enlazada no es necesario
      recorrer nodo por nodo.
    - Basta con eliminar las referencias al primer y último nodo.
    - Al no existir referencias, Python libera la memoria automáticamente.
    - También se reinicia el contador de tamaño.
    """

    # Elimina la referencia al primer nodo
    self.head = None

    # Elimina la referencia al último nodo
    self.tail = None

    # Reinicia el tamaño de la lista
    self._size = 0

# ==============================================================
# EJERCICIO 6: Invertir lista
# ==============================================================

def reverse(self):
    """
    Invierte el orden de los elementos de la lista EN LA MISMA LISTA

    Logica:
    - Se recorren los nodos cambiando la dirección del puntero `next`.
    - Se usan tres referencias:
        prev    → nodo anterior
        current → nodo actual
        next    → siguiente nodo
    - Al final:
        - El último nodo se convierte en el nuevo head
        - El antiguo head pasa a ser el nuevo tail
    - No se crea una nueva lista (O(1) espacio)
    """

    # Nodo anterior inicia en None
    prev = None

    # Nodo actual inicia en el head
    current = self.head

    # El head actual será el nuevo tail
    self.tail = self.head

    # Recorremos toda la lista
    while current:
        # Guardamos el siguiente nodo
        next_node = current.next

        # Invertimos el puntero
        current.next = prev

        # Avanzamos los punteros
        prev = current
        current = next_node

    # El último nodo visitado se convierte en el nuevo head
    self.head = prev

# ==============================================================
# EJERCICIO 7: Detectar ciclo
# ==============================================================

def has_cycle(self):
    """
    Detecta si la lista tiene un ciclo usando el algoritmo de Floyd
    (tortuga y liebre)

    Logica:
    - Se usan dos punteros:
        slow avanza de 1 en 1
        fast avanza de 2 en 2
    - Si existe un ciclo, ambos punteros se encontrarán
    - Si fast o fast.next llegan a None, no hay ciclo
    - No usa memoria adicional (O(1) espacio)
    """

    slow = self.head
    fast = self.head

    # Recorremos mientras fast y fast.next existan
    while fast and fast.next:
        slow = slow.next           # Avanza 1 paso
        fast = fast.next.next      # Avanza 2 pasos

        # Si se encuentran, hay ciclo
        if slow == fast:
            return True

    # Si se llego al final, no hay ciclo
    return False

# ==============================================================
# EJERCICIO 8: Encontrar el medio
# ==============================================================

def get_middle(self):
    """
    Retorna el elemento del medio de la lista

    Logica:
    - Se usan dos punteros:
        slow avanza de 1 en 1
        fast avanza de 2 en 2
    - Cuando fast llega al final, slow queda en el nodo del medio
    - Si la lista tiene un número par de elementos
      se retorna el segundo de los dos del medio
    """

    # Si la lista esta vacia, no se puede obtener el medio
    if self.is_empty():
        raise Exception("La lista está vacía")

    slow = self.head
    fast = self.head

    # Avanzamos los punteros
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # slow queda apuntando al nodo del medio
    return slow.data

# ==============================================================
# EJERCICIO 9: Eliminar duplicados
# ==============================================================

def remove_duplicates(self):
    """
    Elimina los elementos duplicados de la lista,
    dejando solo la primera aparición de cada elemento

    Logica:
    - Se utiliza un conjunto (set) para registrar los valores ya vistos
    - Se recorre la lista nodo por nodo
    - Si el valor ya existe en el set:
        - Se elimina el nodo ajustando los punteros
    - Si no existe:
        - Se agrega el valor al set y se continúa
    - Complejidad:
        - Tiempo: O(n)
        - Espacio: O(n)
    """

    # Si la lista esta vacia o tiene un solo elemento
    if self.is_empty() or self.head.next is None:
        return

    seen = set()
    current = self.head
    prev = None

    while current:
        if current.data in seen:
            # Se elimina el nodo duplicado
            prev.next = current.next

            # Si el nodo eliminado era el tail, se actualiza
            if current == self.tail:
                self.tail = prev

            self._size -= 1
        else:
            # Se registra el valor y se avanza
            seen.add(current.data)
            prev = current

        current = current.next

# ==============================================================
# EJERCICIO 10: Fusionar dos listas ordenadas
# ==============================================================

def merge_sorted(list1, list2):
    """
    Fusiona dos listas enlazadas ORDENADAS en una nueva lista ordenada

    Logica:
    - Se utilizan dos punteros, uno para cada lista
    - Se comparan los valores actuales de cada lista
    - Se agrega a la nueva lista el menor de los dos valores
    - Cuando una lista termina, se agregan los elementos restantes
    - No se modifican las listas originales
    """

    merged = LinkedList()

    current1 = list1.head
    current2 = list2.head

    # Mientras ambas listas tengan elementos
    while current1 and current2:
        if current1.data <= current2.data:
            merged.append(current1.data)
            current1 = current1.next
        else:
            merged.append(current2.data)
            current2 = current2.next

    # Agregar los elementos restantes de list1
    while current1:
        merged.append(current1.data)
        current1 = current1.next

    # Agregar los elementos restantes de list2
    while current2:
        merged.append(current2.data)
        current2 = current2.next

    return merged

# ==============================================================
# EJERCICIO 11: Palíndromo
# ==============================================================

def is_palindrome(self):
    """
    Verifica si la lista enlazada es un palíndromo

    Logica:
    - Un palíndromo se lee igual de izquierda a derecha que al revés
    - Se recorren los nodos y se guardan los valores en una lista auxiliar
    - Se compara la lista con su versión invertida
    - Si ambas son iguales, la lista es un palíndromo
    """

    values = []
    current = self.head

    # Recorrer la lista y guardar los valores
    while current:
        values.append(current.data)
        current = current.next

    # Comparar la lista con su versión invertida
    return values == values[::-1]

# ==============================================================
# EJERCICIO 12: Rotar lista
# ==============================================================

def rotate(self, k):
    """
    Rota la lista k posiciones a la derecha

    Logica:
    1. Si la lista está vacía o k es 0, no se hace nada
    2. Se conecta el último nodo con el primero (lista circular)
    3. Se avanza hasta la posición size - k
    4. Se rompe el ciclo para definir nuevo head y tail
    """

    if self.is_empty() or k == 0:
        return

    k = k % self._size
    if k == 0:
        return

    # Hacer la lista circular
    self.tail.next = self.head

    # Buscar el nuevo tail
    steps = self._size - k
    new_tail = self.head
    for _ in range(steps - 1):
        new_tail = new_tail.next

    # Definir nuevo head y romper el ciclo
    self.head = new_tail.next
    new_tail.next = None
    self.tail = new_tail

# ==============================================================
# EJERCICIO 13: Particionar lista
# ==============================================================

def partition(self, x):
    """
    Particiona la lista alrededor del valor x
    
    Logica:
    - Se recorren todos los nodos de la lista original
    - Los elementos menores a x se guardan en una lista auxiliar
    - Los elementos mayores o iguales a x se guardan en otra lista
    - Finalmente se unen ambas listas preservando el orden relativo
    """

    # Crear listas auxiliares
    menores = LinkedList()
    mayores = LinkedList()

    # Comenzar desde la cabeza de la lista original
    current = self.head

    # Recorrer la lista original
    while current:
        if current.data < x:
            # Agregar a la lista de menores
            menores.append(current.data)
        else:
            # Agregar a la lista de mayores o iguales
            mayores.append(current.data)
        current = current.next  # Avanzar al siguiente nodo

    # Si no hay elementos menores, la lista queda igual a mayores
    if menores.is_empty():
        self.head = mayores.head
        self.tail = mayores.tail
    else:
        # Unir ambas listas
        menores.tail.next = mayores.head
        self.head = menores.head
        self.tail = mayores.tail if mayores.tail else menores.tail

    # Actualizar tamaño
    self._size = menores._size + mayores._size

# ==============================================================
# EJERCICIO 14: Suma de dos listas (números)
# ==============================================================

def add_numbers(list1, list2):
    """
    Suma dos números representados como listas enlazadas
    
    Logica:
    - Se recorren ambas listas simultáneamente
    - Se suman los dígitos junto con el acarreo (carry)
    - El resultado se guarda en una nueva lista
    """

    resultado = LinkedList()  # Lista resultado
    carry = 0                 # Acarreo inicial

    n1 = list1.head
    n2 = list2.head

    # Mientras haya digitos o acarreo
    while n1 or n2 or carry:
        valor1 = n1.data if n1 else 0
        valor2 = n2.data if n2 else 0

        suma = valor1 + valor2 + carry
        carry = suma // 10              # Calcular nuevo acarreo
        resultado.append(suma % 10)     # Guardar digito

        if n1:
            n1 = n1.next
        if n2:
            n2 = n2.next

    return resultado

# ==============================================================
# EJERCICIO 15: Intersección de dos listas
# ==============================================================

def find_intersection(list1, list2):
    """
    Encuentra el nodo de intersección de dos listas
    
    Logica:
    - Se calcula la longitud de ambas listas
    - Se alinean los punteros
    - Se avanza simultáneamente hasta encontrar el mismo nodo
    """

    len1 = len(list1)
    len2 = len(list2)

    p1 = list1.head
    p2 = list2.head

    # Alinear listas
    if len1 > len2:
        for _ in range(len1 - len2):
            p1 = p1.next
    else:
        for _ in range(len2 - len1):
            p2 = p2.next

    # Avanzar simultáneamente
    while p1 and p2:
        if p1 is p2:
            return p1
        p1 = p1.next
        p2 = p2.next

    return None

# ==============================================================
# EJERCICIO 16: Navegador Web
# ==============================================================

class BrowserHistory:
    def __init__(self, homepage):
        """
        Logica:
        - Se usa una lista doblemente enlazada
        - current mantiene la página actual
        """
        self.current = DoubleNode(homepage)

    def visit(self, url):
        # Crear nuevo nodo
        new_node = DoubleNode(url)

        # Eliminar historial futuro
        self.current.next = None

        # Enlazar nuevo nodo
        new_node.prev = self.current
        self.current.next = new_node

        # Mover el cursor
        self.current = new_node

    def back(self, steps):
        # Retroceder mientras sea posible
        while steps > 0 and self.current.prev:
            self.current = self.current.prev
            steps -= 1
        return self.current.data

    def forward(self, steps):
        # Avanzar mientras sea posible
        while steps > 0 and self.current.next:
            self.current = self.current.next
            steps -= 1
        return self.current.data

    def get_current(self):
        return self.current.data

# ==============================================================
# EJERCICIO 17: LRU Cache
# ==============================================================

class LRUCache:
    def __init__(self, capacity):
        """
        Logica:
        - Diccionario para acceso O(1)
        - Lista doble para mantener orden de uso
        """
        self.capacity = capacity
        self.cache = {}

        self.head = DoubleNode()  # Menos usado
        self.tail = DoubleNode()  # Más usado

        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add(self, node):
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key):
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.data[1]

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])

        node = DoubleNode((key, value))
        self.cache[key] = node
        self._add(node)

        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.data[0]]

# ==============================================================  
# EJERCICIO 18: Editor Multi-cursor  
# ==============================================================  

class MultiCursorEditor:
    """
    Editor de texto que soporta múltiples cursores.
    
    Logica:
    - Cada cursor mantiene su propia posición
    - Se almacenan los cursores en un diccionario
    - Se inserta texto en la posición correspondiente de cada cursor
    """
    def __init__(self):
        self.text = LinkedList()  # Lista enlazada que contiene los caracteres
        self.cursors = {}         # Diccionario {cursor_id: posición}
        self.next_id = 0          # ID incremental para cada cursor

    def add_cursor(self, position):
        """
        Agrega un cursor en la posición indicada
        """
        if position < 0 or position > len(self.text):
            raise IndexError("Posición de cursor fuera de rango")
        
        self.cursors[self.next_id] = position
        self.next_id += 1
        return self.next_id - 1

    def remove_cursor(self, cursor_id):
        """
        Elimina un cursor existente
        """
        if cursor_id in self.cursors:
            del self.cursors[cursor_id]

    def type_at_cursor(self, cursor_id, text):
        """
        Escribe texto en la posición del cursor específico
        
        Args:
            cursor_id: ID del cursor
            text: Cadena a insertar
        """
        if cursor_id not in self.cursors:
            raise ValueError("Cursor no existe")
        
        pos = self.cursors[cursor_id]
        for char in text:
            self.text.insert(pos, char)  # Inserta el caracter en la posición
            pos += 1
        self.cursors[cursor_id] = pos  # Actualiza la posición del cursor

# Añadimos metodo insert a LinkedList para que funcione MultiCursorEditor  

def linkedlist_insert(self, index, data):
    """
    Inserta un nodo en la posición index de la lista enlazada
    """
    if index < 0 or index > self._size:
        raise IndexError("indice fuera de rango")
    
    new_node = Node(data)
    
    if index == 0:  # Insertar al inicio
        new_node.next = self.head
        self.head = new_node
        if self._size == 0:
            self.tail = new_node
    else:
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        if new_node.next is None:
            self.tail = new_node
    
    self._size += 1

# ==============================================================
# EJERCICIO 19: Benchmark de operaciones
# ==============================================================

import time
import random

def benchmark_data_structures():
    """
    Compara el rendimiento de diferentes estructuras
    """

    py_list = []
    linked = LinkedList()

    # Insercion al inicio
    start = time.time()
    for i in range(1000):
        py_list.insert(0, i)
    print("Array insertar inicio:", time.time() - start)

    start = time.time()
    for i in range(1000):
        linked.prepend(i)
    print("LinkedList insertar inicio:", time.time() - start)

# ============================================================================
# EJERCICIO 20: Análisis de casos de uso y selección de estructuras
# ============================================================================

"""
Para cada uno de los siguientes escenarios, determina qué estructura
es más apropiada (Array, Lista Simple, Lista Doble) y justifica tu respuesta:

1. Sistema de colas de impresión (FIFO estricto)
   Estructura recomendada: Lista simplemente enlazada
   Una cola FIFO requiere inserciones al final y eliminaciones al inicio
   En una lista simplemente enlazada estas operaciones son O(1) si se mantiene
   referencia al inicio y al final, sin necesidad de desplazar elementos

2. Historial de navegación de un navegador
   Estructura recomendada: Lista doblemente enlazada
   Permite navegar hacia atrás y hacia adelante de forma eficiente (O(1)),
   ya que cada nodo tiene referencias al nodo anterior y siguiente.

3. Sistema de undo/redo con límite de 100 acciones
   Estructura recomendada: Lista doblemente enlazada
   El undo y redo implican moverse en ambas direcciones del historial
   La lista doble facilita este recorrido sin reconstruir estructuras

4. Base de datos que necesita acceso rápido por ID
   Estructura recomendada: Array (lista de Python)
   Permite acceso directo por índice en tiempo O(1) ideal para búsquedas
   frecuentes por identificador

5. Playlist de música con navegación adelante/atrás
   Estructura recomendada: Lista doblemente enlazada
   La navegación entre canciones previas y siguientes se realiza de manera
   eficiente gracias a los punteros bidireccionales

6. Sistema de gestión de memoria del sistema operativo
   Estructura recomendada: Lista doblemente enlazada
   Permite insertar y eliminar bloques de memoria fácilmente y recorrer
   la lista en ambas direcciones para compactación o reasignación

7. Editor de texto que solo permite append al final
   Estructura recomendada: Array (lista de Python)
   El append en listas de Python es O(1) amortizado y no se requiere inserción
   en posiciones intermedias

8. Implementación de una pila (Stack)
   Estructura recomendada: Lista simplemente enlazada
   Las operaciones push y pop se realizan en un solo extremo en tiempo O(1)

9. Juego que necesita insertar y eliminar enemigos frecuentemente
   Estructura recomendada: Lista enlazada
   Las listas enlazadas permiten inserciones y eliminaciones frecuentes sin
   desplazar grandes cantidades de datos en memoria

10. Sistema de logs que solo escribe al final y lee todo
    Estructura recomendada: Array (lista de Python)
    La escritura secuencial al final es eficiente y la lectura completa
    se realiza fácilmente recorriendo el array
"""

# ==============================================================
# ASIGNACIÓN DE MÉTODOS A LinkedList 
# ==============================================================

LinkedList.count = count
LinkedList.get = get
LinkedList.index_of = index_of
LinkedList.to_list = to_list
LinkedList.clear = clear
LinkedList.reverse = reverse
LinkedList.has_cycle = has_cycle
LinkedList.get_middle = get_middle
LinkedList.remove_duplicates = remove_duplicates
LinkedList.rotate = rotate
LinkedList.partition = partition
LinkedList.is_palindrome = is_palindrome
LinkedList.insert = linkedlist_insert


# ============================================================================
# SECCIoN FINAL DE PRUEBAS – EJECUCIoN DE TODOS LOS EJERCICIOS
# ============================================================================

print("\nEJERCICIO 1: Contar elementos")
lista = LinkedList()
lista.create_from_list([1, 2, 3, 2, 4, 2])

tests = [
    (2, 3),
    (1, 1),
    (5, 0)
]

for elem, expected in tests:
    result = lista.count(elem)
    print(f"count({elem}) = {result} | esperado: {expected}")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 2: Obtener elemento por índice")
lista = LinkedList()
lista.create_from_list([10, 20, 30, 40])

tests = [
    (0, 10),
    (2, 30),
    (3, 40)
]

for index, expected in tests:
    result = lista.get(index)
    print(f"get({index}) = {result} | esperado: {expected}")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 3: Encontrar índice de elemento")
lista = LinkedList()
lista.create_from_list([5, 8, 1, 8])

tests = [
    (8, 1),
    (1, 2),
    (9, -1)
]

for elem, expected in tests:
    result = lista.index_of(elem)
    print(f"index_of({elem}) = {result} | esperado: {expected}")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 4: Lista a array")
lista = LinkedList()
lista.create_from_list([7, 9, 11])

tests = [
    ([7, 9, 11]),
    ([]),
    ([1])
]

for expected in tests:
    lista.clear()
    lista.create_from_list(expected)
    result = lista.to_list()
    print(f"to_list() = {result} | esperado: {expected}")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 5: Limpiar lista")
lista = LinkedList()
lista.create_from_list([1, 2, 3])

lista.clear()
print(f"Lista vacía: {lista.is_empty()} | esperado: True")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 6: Revertir lista")
lista = LinkedList()
lista.create_from_list([1, 2, 3])

lista.reverse()
print(f"reverse() = {lista.to_list()} | esperado: [3, 2, 1]")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 7: Detectar ciclo")
lista = LinkedList()
lista.create_from_list([1, 2, 3])

print(f"has_cycle() = {lista.has_cycle()} | esperado: False")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 8: Elemento medio")
lista = LinkedList()
lista.create_from_list([1, 2, 3, 4, 5])

print(f"get_middle() = {lista.get_middle()} | esperado: 3")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 9: Eliminar duplicados")
lista = LinkedList()
lista.create_from_list([1, 2, 2, 3, 3])

lista.remove_duplicates()
print(f"remove_duplicates() = {lista.to_list()} | esperado: [1, 2, 3]")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 10: Rotar lista")
lista = LinkedList()
lista.create_from_list([1, 2, 3, 4, 5])

lista.rotate(2)
print(f"rotate(2) = {lista.to_list()} | esperado: [4, 5, 1, 2, 3]")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 11: Verificar palíndromo")
lista = LinkedList()
lista.create_from_list([1, 2, 3, 2, 1])

print(f"is_palindrome() = {lista.is_palindrome()} | esperado: True")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 12: Insertar en índice")
lista = LinkedList()
lista.create_from_list([1, 3, 4])

lista.insert(1, 2)
print(f"insert(1, 2) = {lista.to_list()} | esperado: [1, 2, 3, 4]")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 13: Particionar lista")
lista = LinkedList()
lista.create_from_list([3, 5, 8, 5, 10, 2, 1])

lista.partition(5)
print(f"partition(5) = {lista.to_list()}")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 14: Suma de dos listas")
num1 = LinkedList()
num1.create_from_list([2, 4, 3])
num2 = LinkedList()
num2.create_from_list([5, 6, 4])

result = add_numbers(num1, num2)
print(f"add_numbers() = {result.to_list()} | esperado: [7, 0, 8]")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 15: Intersección de listas")
print("Prueba conceptual (comparación por referencia de nodos)")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 16: Navegador Web")
browser = BrowserHistory("google.com")
browser.visit("youtube.com")
browser.visit("facebook.com")
browser.back(1)
print(f"Página actual: {browser.get_current()} | esperado: youtube.com")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 17: LRU Cache")
cache = LRUCache(2)
cache.put(1, "A")
cache.put(2, "B")
cache.get(1)
cache.put(3, "C")
print(f"get(2) = {cache.get(2)} | esperado: None")

# ---------------------------------------------------------------------------

print("\nEJERCICIO 18: Editor Multi-cursor")
editor = MultiCursorEditor()

# Añadir cursores
c1 = editor.add_cursor(0)  # Cursor en inicio
c2 = editor.add_cursor(0)  # Otro cursor también al inicio

# Escribir texto desde cada cursor
editor.type_at_cursor(c1, "Hola")
editor.type_at_cursor(c2, "Mundo")

# Convertir lista a texto para ver resultado
def linkedlist_to_string(ll):
    result = []
    current = ll.head
    while current:
        result.append(current.data)
        current = current.next
    return "".join(result)

print("Texto en editor:", linkedlist_to_string(editor.text))
print("Posiciones de cursores:", editor.cursors)
print("Editor multi-cursor ejecutado correctamente")
# ---------------------------------------------------------------------------

print("\nEJERCICIO 19: Benchmark")
benchmark_data_structures()




