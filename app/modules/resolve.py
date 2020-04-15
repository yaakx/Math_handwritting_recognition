import numpy as np
from sympy import Symbol, lambdify, Eq, solveset, S
from sympy.solvers import solve
from abc import ABC, abstractmethod


class MathTypes(ABC):
    def __init__(self, equation, symbol=None):
        self.equation = equation
        self.symbol = symbol
        self.solution = self.resolve()
        self.str_sol = self.to_str()

    @abstractmethod
    def resolve(self):
        pass

    @abstractmethod
    def to_str(self):
        return ""

    def __str__(self):
        return self.str_sol

    def __repr__(self):
        return (f'{self.__class__.__name__}('
           f'{self.solution!r})')


class Calculator(MathTypes):
    def resolve(self):
        self.solution = eval(self.equation)
        return self.solution

    def to_str(self):
        return f'[{str(self.solution)}]'


class Equation(MathTypes):
    def resolve(self):
        eq = f'{self.equation.split("=")[0]}-({self.equation.split("=")[1]})'
        eq = eq.replace(self.symbol[0], 'x')
        x = Symbol('x')
        self.solution = solve(eval(eq), x)
        return self.solution

    def to_str(self):
        s = '['
        for i in range(len(self.solution)):
            s += f'{self.symbol[0]}={str(self.solution[i])}'
            if i != len(self.solution) - 1:
                s += ","
        s += ']'
        return s


class Function(MathTypes):
    def resolve(self):
        if len(self.symbol) < 3:
            eq = f'{self.equation.split("=")[0]}-({self.equation.split("=")[1]})'
            if 'x' in self.symbol:
                eq = eq.replace('z', 'y')
            else:
                eq.replace('z', 'x')
            x = Symbol('x')
            y = Symbol('y')
            f = eval(eq)
            f_l = lambdify(x, f)
            self.solution = []
            lin = np.linspace(-20, 20, 50)
            print(f)
            for i in lin:
                try:
                    x_sol = solve(f_l(i), y)
                    for j in x_sol:
                        self.solution.append([i, j])
                except:
                    pass
        return self.solution

    def derivative(self, symbol):
        if 'x' in self.symbol:
            eq = self.equation.replace('z', 'y')
        else:
            eq = self.equation.replace('z', 'x')
        x = Symbol('x')
        y = Symbol('y')
        f = Eq(eval(eq.split("=")[0]), eval(eq.split("=")[1]))
        try:
            sol = solveset(f, symbol, domain=S.Reals)
        except:
            return None

    def to_str(self):
        return str(self.equation)


class MathSolver(object):
    def __init__(self, equation):
        self.equation = equation
        self.type = None
        self.solution = self.classifier()

    def classifier(self):
        if '=' not in self.equation:
            self.type = 'C'
            return Calculator(self.equation)
        elif ('x' in self.equation and 'y' not in self.equation and 'z' not in self.equation) or \
                ('y' in self.equation and 'x' not in self.equation and 'z' not in self.equation) or \
                ('z' in self.equation and 'y' not in self.equation and 'x' not in self.equation):
            self.type = 'E'
            return Equation(self.equation, ['x'] if 'x' in self.equation else ['y'] if 'y' in self.equation else ['z'])
        else:
            symbol = []
            for i in self.equation:
                if i == 'x' or i == 'y' or i == 'z':
                    symbol.append(i)
            self.type = 'F'
            return Function(self.equation, list(set(symbol)))

    def __str__(self):
        return str(self.solution)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
           f'{self.equation!r}, {self.solution!r})')


if __name__ == "__main__":
    import os
    from app.modules.my_class import ImageSolver

    l = os.listdir("../data/images")
    print(l)
    d = ImageSolver(f'../data/images/{l[-6]}', "../data/models/third_model.h5")
    m = MathSolver(d.equation)
    print(m.solution)
    """print(repr(d))
    print(d)
    print(m)
    print(repr(m))
    print(m.solution)"""
