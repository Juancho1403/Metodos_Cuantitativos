import numpy as np

class ProblemaLineal:
    def __init__(self, c, A, b, indices_enteros=None):
        """
        c: Coeficientes de la función objetivo (maximizar)
        A: Matriz de restricciones (<=)
        b: Vector de términos independientes
        indices_enteros: lista de índices de variables que deben ser enteras
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.indices_enteros = indices_enteros or []

class SolucionSimplex:
    def __init__(self, problema: ProblemaLineal):
        self.problema = problema

    def resolver(self):
        m, n = self.problema.A.shape
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = self.problema.A
        tableau[:m, n:n + m] = np.eye(m)
        tableau[:m, -1] = self.problema.b
        tableau[-1, :n] = -self.problema.c
        base = list(range(n, n + m))

        while True:
            if all(x >= 0 for x in tableau[-1, :-1]):
                break
            col = np.argmin(tableau[-1, :-1])
            ratios = [tableau[i, -1] / tableau[i, col] if tableau[i, col] > 1e-8 else np.inf for i in range(m)]
            fila = np.argmin(ratios)
            if ratios[fila] == np.inf:
                raise Exception("Problema ilimitado")
            pivote = tableau[fila, col]
            tableau[fila] /= pivote
            for i in range(m + 1):
                if i != fila:
                    tableau[i] -= tableau[i, col] * tableau[fila]
            base[fila] = col

        x = np.zeros(n + m)
        for i, bcol in enumerate(base):
            x[bcol] = tableau[i, -1]
        valor = tableau[-1, -1]
        return x[:n], valor, tableau, base

class SolucionPlanosDeCorte:
    def __init__(self, problema: ProblemaLineal):
        self.problema_inicial = problema

    def _corte_gomory(self, tableau, base):
        m, _ = tableau.shape
        filas_fracc = [(i, tableau[i, -1] % 1) for i in range(m - 1) if tableau[i, -1] % 1 > 1e-8]
        if not filas_fracc:
            return None
        fila, _ = max(filas_fracc, key=lambda x: x[1])
        fila_tab = tableau[fila]
        coef_fracc = -(fila_tab[:-1] % 1)
        termino_fracc = -(fila_tab[-1] % 1)
        return coef_fracc, termino_fracc

    def resolver(self):
        prob = ProblemaLineal(
            self.problema_inicial.c.copy(),
            self.problema_inicial.A.copy(),
            self.problema_inicial.b.copy(),
            self.problema_inicial.indices_enteros
        )

        while True:
            simplex = SolucionSimplex(prob)
            x_relajado, valor, tableau, base = simplex.resolver()
            print(f"Solución relajada: x={x_relajado}, valor={valor}")
            if all(np.isclose(x_relajado[i], np.floor(x_relajado[i])) for i in prob.indices_enteros):
                return x_relajado, valor
            corte = self._corte_gomory(tableau, base)
            if corte is None:
                break
            coef, termino = corte
            prob.A = np.vstack([prob.A, coef])
            prob.b = np.append(prob.b, termino)
            print(f"Añadido corte de Gomory: {coef} <= {termino}")

# Ejemplo de un problema del mundo real:
# Una empresa de reparto dispone de 3 camiones y debe asignarlos a 2 rutas diarias.
# Cada camión en la Ruta 1 genera un beneficio de 400 USD, en la Ruta 2 de 550 USD.
# La Ruta 1 requiere 2 horas de conducción y la Ruta 2 requiere 3 horas.
# Cada camión dispone de un máximo de 8 horas diarias.
# La empresa necesita que cada ruta sea atendida por al menos 2 camiones (enteros).
# Variables: x0 = nº camiones en Ruta 1, x1 = nº camiones en Ruta 2.
# Max z = 400·x0 + 550·x1
# s.a.
#   2·x0 + 3·x1 ≤ 8·3   (horas totales disponibles)
#   x0 ≥ 2
#   x1 ≥ 2
# x0, x1 enteros
# Se reescribe ≥ como ≤ multiplicando por -1.

if __name__ == '__main__':
    print("### Método de Planos de Corte ###")
    print("Problema: Asignación de camiones a rutas de reparto")
    n = 2  # x0, x1
    m = 3  # restricciones: horas, mínimo x0, mínimo x1
    c = [400, 550]
    A = [ [2, 3],    # 2x0+3x1 <= 24
          [-1, 0],   # -x0 <= -2  => x0 >= 2
          [0, -1] ]  # -x1 <= -2  => x1 >= 2
    b = [24, -2, -2]
    indices = [0, 1]
    problema = ProblemaLineal(c, A, b, indices)
    solver = SolucionPlanosDeCorte(problema)
    solucion, valor_opt = solver.resolver()
    print("\nSolución entera óptima (nº camiones):", solucion)
    print("Beneficio máximo (USD):", valor_opt)
    print("### Fin del programa ###")