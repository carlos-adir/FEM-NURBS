

import numpy as np
import femnurbs.SplineUsefulFunctions as SUF

class BaseSpline:

    def __doc__(self):
        """
        This function is recursivly determined like

        N_{i, 0}(u) = { 1   if  U[i] <= u < U[i+1]
                      { 0   else

                          u - U[i]
        N_{i, j}(u) = --------------- * N_{i, j-1}(u)
                       U[i+j] - U[i]
                            U[i+j+1] - u
                      + ------------------- * N_{i+1, j-1}(u)
                         U[i+j+1] - U[i+1]

        As consequence, we have that

        N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )


        """
        pass

    @staticmethod
    def compute_np(U):
        """

        We have that U = [0, ..., 0, ?, ..., ?, 1, ..., 1]
        And that U[p] = 0, but U[p+1] != 0
        The same way, U[n] = 1, but U[n-1] != 0

        Using that, we know that
            len(U) = m + 1 = n + p + 1
        That means that
            m = n + p

        """
        m = len(U) - 1
        p = -1
        while U[p + 1] == 0 and U[m - (p + 1)] == 1:
            p += 1
        if U[p + 1] == 0 or U[m - (p + 1)] == 1:
            msg = "The vector U is not valid. Problem while calculing p"
            raise Exception(msg)
        n = m - p
        return n, p
        pass

    @staticmethod
    def validate_entry_U(U):
        SUF.isValidU(U, throwError=True)

    def __init__(self, U):
        """

        We have that the object can be called like
        N = SplineBaseFunction(U)

        And so, to get the value of N_{i, j}(u) can be express like
        N[i, j](u)

        """
        BaseSpline.validate_entry_U(U)
        n, p = BaseSpline.compute_np(U)
        self.__U = np.copy(np.array(U))
        self.__p = p
        self.__n = n
        self.__computeDivisions()

    def __computeDivisions(self):
        self.__div = np.zeros((self.n + 1, self.p + 1))
        for i in range(self.n + 1):
            for j in range(1, self.p + 1):
                if self.U[i + j] != self.U[i]:
                    self.__div[i, j] = 1 / (self.U[i + j] - self.U[i])

    def __find_spot(self, u):
        if isinstance(u, np.ndarray):
            k = np.ones(u.shape, dtype="int64")
            for iii in range(u.shape[0]):
                k[iii] = self.__find_spot(u[iii])
            return k
        else:
            if u == 0:
                return self.p
            if u == 1:
                return self.n - 1
            lower, upper = self.p, self.n
            mid = (lower + upper) // 2
            while True:
                if u < self.U[mid]:
                    upper = mid
                elif self.U[mid + 1] <= u:
                    lower = mid
                else:
                    return mid
                mid = (lower + upper) // 2

    def __compute_N_withspot(self, i, j, k, u):
        """
        returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]

        We have that N_{ij}(u) = 0 if  u not in [u_{i}, u_{i+j+1})

        """
        if k < i or i + j < k:
            return 0
        if j == 0:  # That implies that i = k
            return 1

        factor1 = (u - self.U[i]) * self.div[i, j]
        factor1 *= self.__compute_N_withspot(i, j - 1, k, u)
        factor2 = (self.U[i + j + 1] - u) * self.div[i + 1, j]
        factor2 *= self.__compute_N_withspot(i + 1, j - 1, k, u)
        return factor1 + factor2

    def compute_N(self, i, j, u):
        if not isinstance(u, np.ndarray):
            k = self.__find_spot(u)
            return self.__compute_N_withspot(i, j, k, u)
        else:
            retorno = np.zeros(u.shape)
            for ii in range(u.shape[0]):
                retorno[ii] = self.compute_N(i, j, u[ii])
            return retorno

    @property
    def p(self):
        return self.__p

    @property
    def n(self):
        return self.__n

    @property
    def U(self):
        return self.__U

    @property
    def div(self):
        return self.__div


# def compute_value_N(i, j, k, u, U):
#     """
#     returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]

#     We have that N_{ij}(u) = 0 if  u not in [u_{i}, u_{i+j+1})

#     """
#     p = np.sum(U == 0)

#     if i == n - 1

#     if u == U[-1] and j == 0:
#         if k == n + 1 and i == n:
#             return 1
#         else:
#             return 0
#     elif j == 0:
#         if i == k:
#             return 1
#         else:
#             return 0
#     else:
#         if i + j >= len(U):
#             factor1 = 0
#         elif U[i] == U[i + j]:
#             factor1 = 0
#         else:
#             factor1 = (u - U[i]) / (U[i + j] - U[i])
#         if i + j + 1 >= len(U):
#             factor2 = 0
#         elif U[i + j + 1] == U[i + 1]:
#             factor2 = 0
#         else:
#             factor2 = (U[i + j + 1] - u) / (U[i + j + 1] - U[i + 1])
#         return factor1 * N(i, j - 1, k, u, U) + factor2 * N(i + 1, j - 1, k, u, U)

class CallableFunctionSpline:

    def __init__(self, lines, columns, U):
        self.BS = BaseSpline(U)
        if isinstance(lines, tuple) or isinstance(lines, list):
            lines = np.array(lines, dtype="int64")
        if isinstance(columns, tuple) or isinstance(columns, list):
            columns = np.array(columns, dtype="int64")
        self.__lines = lines
        self.__columns = columns

        self.shape = []
        if isinstance(self.__lines, np.ndarray):
            self.shape.append(len(self.__lines))
        if isinstance(self.__columns, np.ndarray):
            self.shape.append(len(self.__columns))

    def __call__(self, u):
        if isinstance(u, tuple) or isinstance(u, list):
            u = np.array(u)
        shape_to_create = [c for c in self.shape]
        if isinstance(u, np.ndarray):
            for d in u.shape:
                shape_to_create.append(d)
        if len(shape_to_create) == 0:
            retorno = 0
        else:
            retorno = np.zeros(shape_to_create)

        cond1 = isinstance(self.__lines, np.ndarray)
        cond2 = isinstance(self.__columns, np.ndarray)
        if cond1 and cond2:
            for i, line in enumerate(self.__lines):
                for j, column in enumerate(self.__columns):
                    retorno[i, j] += self.BS.compute_N(line, column, u)
        elif cond1 and not cond2:
            column = self.__columns
            for i, line in enumerate(self.__lines):
                retorno[i] += self.BS.compute_N(line, column, u)
        elif not cond1 and cond2:
            line = self.__lines
            for j, column in enumerate(self.__columns):
                retorno[j] += self.BS.compute_N(line, column, u)
        else:
            line = self.__lines
            column = self.__columns
            retorno = self.BS.compute_N(line, column, u)
        return retorno


class SplineBaseFunction(BaseSpline):

    def __init__(self, U):
        """

        We have that the object can be called like
        N = SplineBaseFunction(U)

        And so, to get the value of N_{i, j}(u) can be express like
        N[i, j](u)

        """
        super().__init__(U)

    def __getitem__(self, tup):
        if isinstance(tup, tuple):
            if len(tup) > 2:
                raise IndexError("The dimension of index is maximum 2")
            lines, columns = tup

        else:
            lines = tup
            columns = self.p

        if isinstance(lines, slice):
            if lines != slice(None, None, None):
                raise NotImplementedError("Will be there in the future")
            lines = np.arange(0, self.n, dtype="int64")
        elif isinstance(lines, int):
            if lines < 0:
                raise IndexError("The frist therm must be positive")
            if lines >= self.n:
                raise IndexError("The frist therm must be < " + str(self.n))
        elif isinstance(lines, np.ndarray):
            if np.any(lines < 0):
                raise IndexError("The lines must be positive")
            if np.any(lines >= self.n):
                raise IndexError("The lines must be < " + str(self.n))
        else:
            raise Exception("Not expected get here!! 1234")

        if isinstance(columns, int):
            if columns < 0:
                raise IndexError("The second therm must be positive")
            if columns > self.p:
                raise IndexError("The second therm must be <= " + str(self.p))
        elif isinstance(columns, np.ndarray):
            if np.any(columns < 0):
                raise IndexError("The columns must be positive")
            if np.any(columns > self.p):
                raise IndexError("The columns must be <= " + str(self.p))
        else:
            raise Exception("Not expected get here!! 1235")

        return CallableFunctionSpline(lines, columns, self.U)

    def __call__(self, u):
        lines = np.arange(0, self.n, dtype="int64")
        f = CallableFunctionSpline(lines, self.p, self.U)
        return f(u)

    def __one_over_du_vector(self, j):
        p = self.p
        n = self.n
        U = self.U
        return 1 / (U[p + 1: n + j] - U[p - j + 1: n])

    def get_DerivativeVector(self, j):
        # len(U) = n+p+1
        if j == 0:
            raise ValueError("The value of the parameter must be != 0")
        if j > self.p:
            raise ValueError("The input value must be <= p")
        return j * self.__one_over_du_vector(j)


if __name__ == "__main__":
    pass
