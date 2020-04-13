import matplotlib.pyplot as plt
from sympy import Symbol
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
        return self.equation

    def to_str(self):
        return self.equation


class MathSolver(object):
    def __init__(self, equation, image):
        self.equation = equation
        self.solution = self.classifier()
        self.image = image

    def classifier(self):
        if '=' not in self.equation:
            return Calculator(self.equation)
        elif ('x' in self.equation and 'y' not in self.equation and 'z' not in self.equation) or \
                ('y' in self.equation and 'x' not in self.equation and 'z' not in self.equation) or \
                ('z' in self.equation and 'y' not in self.equation and 'x' not in self.equation):
            return Equation(self.equation, ['x'] if 'x' in self.equation else ['y'] if 'y' in self.equation else ['z'])
        else:
            return Function(self.equation)

    def image_show(self):
        plt.imshow(self.image, cmap='gray')
        plt.show()

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
    d = ImageSolver(f'../data/images/{l[1]}', "../data/models/third_model.h5")
    m = MathSolver(d.equation, d.numbers.white)
    print(repr(d))
    print(d)
    print(m)
    print(repr(m))
    print(m.solution)
