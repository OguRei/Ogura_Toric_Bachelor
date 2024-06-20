import numpy as np
from enum import IntEnum, auto
import random
import torch
import torch.nn.functional as F
import toric_param as toric_param


class Pauli(IntEnum):
    I = auto()
    X = auto()
    Y = auto()
    Z = auto()

    # パウリ行列の掛け算
    def __mul__(self, b):
        if self == b:
            return Pauli.I
        if self is Pauli.I:
            return b
        if b is Pauli.I:
            return self
        return ({Pauli.X, Pauli.Y, Pauli.Z} - {self, b}).pop()

        # 単にどのパウリ行列かを返すため
        #   def __str__(self) -> str:
        if self == Pauli.I:
            return "I"
        if self == Pauli.X:
            return "X"
        if self == Pauli.Y:
            return "Y"
        if self == Pauli.Z:
            return "Z"
        # エラー時に吐く
        raise ValueError("invalid value")


class ToricCode:
    def __init__(self, param) -> None:
        self.size = param.code_distance
        self.errors_rate = param.errors_rate

    def generate_errors(self) -> torch.Tensor:
        errors_list = random.choices(
            [0, 1, 2, 3],  # 0=I, 1=X, 2=Y, 3=Z
            [
                1 - self.errors_rate[0] - self.errors_rate[1] - self.errors_rate[2],
                self.errors_rate[0],
                self.errors_rate[1],
                self.errors_rate[2],
            ],
            k=2 * self.size * self.size,
        )
        # 2n*nの辺上の量子ビットを表すk.nは符号距離
        errors = torch.Tensor(errors_list)
        errors = errors.reshape(2 * self.size, self.size)
        return errors
    
    def errors_reshape(self, errors) -> torch.Tensor:
        errors = errors
        errors_reshape = errors.reshape(1,1,self.size*2, self.size)
        return errors_reshape

    # def generate_errors_with_n(self):

    def generate_syndrome_Z(self, errors) -> torch.Tensor:  # 面
        syndZ = torch.zeros((self.size, self.size),dtype = torch.float32)
        for i in range(self.size):  # 行
            for j in range(self.size):  # 列
                count_Z_operator = 0
                if j == self.size - 1:  # 右
                    if errors[2 * i + 1][0] == 1 or errors[2 * i + 1][0] == 2:
                        count_Z_operator += 1
                else:
                    if errors[2 * i + 1][j + 1] == 1 or errors[2 * i + 1][j + 1] == 2:
                        count_Z_operator += 1
                if i == self.size - 1:  # 下
                    if errors[0][j] == 1 or errors[0][j] == 2:
                        count_Z_operator += 1
                else:
                    if errors[2 * i + 2][j] == 1 or errors[2 * i + 2][j] == 2:
                        count_Z_operator += 1
                if errors[2 * i + 1][j] == 1 or errors[2 * i + 1][j] == 2:  # 左
                    count_Z_operator += 1
                if errors[2 * i][j] == 1 or errors[2 * i][j] == 2:  # 上
                    count_Z_operator += 1

                # Z周りの４つのqubitを確認
                syndZ[i, j] = 0.5 * (1 - torch.cos(torch.tensor(np.pi) * count_Z_operator))
                
        return syndZ
    
    def generate_syndrome_Z_differentiable(self, errors) -> torch.Tensor:
        syndZ_list = []
        for b in errors: 
            syndZ = torch.zeros((self.size, self.size), dtype=torch.float32)
            errors_b = b[0,:,:]
            for i in range(self.size):
                for j in range(self.size):
                    count_Z_operator = 0.0
                    if j == self.size - 1:  # 右
                        count_Z_operator += errors_b[2 * i + 1][0]
                    else:
                        count_Z_operator += errors_b[2 * i + 1][j + 1]
                    if i == self.size - 1:  # 下
                        count_Z_operator += errors_b[0][j]
                    else:
                        count_Z_operator += errors_b[2 * i + 2][j]
                    
                    count_Z_operator += errors_b[2 * i + 1][j]  # 左
                    count_Z_operator += errors_b[2 * i][j]  # 上


                    syndZ[i, j] = 0.5 * (1 - torch.cos(torch.tensor(np.pi) * count_Z_operator))
            syndZ_list.append(syndZ)
                    
        return torch.stack(syndZ_list)

    def generate_syndrome_X(self, errors) -> torch.Tensor:
        syndX = torch.zeros((self.size, self.size), dtype=torch.float32)
        for i in range(self.size):
            for j in range(self.size):
                count_X_operator = 0
                if j == 0:  # 左
                    if errors[2 * i][self.size - 1] == 2 or errors[2 * i][self.size - 1] == 3:
                        count_X_operator += 1
                else:
                    if errors[2 * i][j - 1] == 2 or errors[2 * i][j - 1] == 3:
                        count_X_operator += 1
                if i == 0:  # 上
                    if errors[2 * self.size - 1][j] == 2 or errors[2 * self.size - 1][j] == 3:
                        count_X_operator += 1
                else:
                    if errors[2 * i - 1][j] == 2 or errors[2 * i - 1][j] == 3:
                        count_X_operator += 1
                if errors[2 * i + 1][j] == 2 or errors[2 * i + 1][j] == 3:  # 下
                    count_X_operator += 1
                if errors[2 * i][j] == 2 or errors[2 * i][j] == 3:  # 右
                    count_X_operator += 1

                # X周りの４つのqubitを確認
                syndX[i, j] = 0.5 * (1 - torch.cos((torch.tensor(np.pi) * count_X_operator)))
                
        return syndX
    
    def generate_syndrome_X_differentiable(self, errors) -> torch.Tensor:
        syndX_list = []
        for b in errors:
            syndX = torch.zeros((self.size, self.size), dtype=torch.float32)
            errors_b = b[0,:,:]
            for i in range(self.size):
                for j in range(self.size):
                    count_X_operator = 0.0
                    if j == 0:
                        count_X_operator += errors_b[2 * i][self.size - 1]
                    else:
                        count_X_operator += errors_b[2 * i][j - 1]
                    if i == 0:
                        count_X_operator += errors_b[2 * self.size - 1][j]
                    else:
                        count_X_operator += errors_b[2 * i - 1][j]
                    
                    count_X_operator += errors_b[2 * i + 1][j]
                    count_X_operator += errors_b[2 * i][j]

                    syndX[i, j] = 0.5 * (1 - torch.cos(torch.tensor(np.pi) * count_X_operator))
            syndX_list.append(syndX)
        return torch.stack(syndX_list)


    def not_has_non_trivial_X(self, errors) -> bool:
        errors_temp = torch.zeros((2, self.size, self.size),dtype = int)
        i, j, m, n = 0, 0, 0, 0
        while i < 2 * self.size:
            while j < self.size:
                errors_temp[0, m, n] = errors[i, j]
                j = j + 1
                n = n + 1
            n = 0
            j = 0
            i = i + 2
            m = m + 1
        
        j, m, n = 0,0,0
        i = 1
        while i < 2 * self.size:
            while j < self.size:
                errors_temp[1, m, n] = errors[i, j]
                j = j + 1
                n = n + 1
            n = 0
            j = 0
            i = i + 2
            m = m + 1
        
        count = [0, 0]
        for i in range(self.size):
            for j in range(self.size):
                if errors_temp[0, i, j] == 1 or errors_temp[0, i, j] == 2:
                    count[0] += 1
        
        for i in range(self.size):
            for j in range(self.size):
                if errors_temp[1, i, j] == 1 or errors_temp[1, i, j] == 2:
                    count[1] += 1
        #print(count)
        return count[0] % 2 == 0 and count[1] % 2 == 0

    def not_has_non_trivial_Z(self, errors) -> bool:
        errors_temp = torch.zeros((2, self.size, self.size), dtype=int)
        i, j, m, n = 0, 0, 0, 0
        while i < 2 * self.size:
            while j < self.size:
                errors_temp[0, m, n] = errors[i, j]
                j = j + 1
                n = n + 1
            n = 0
            j = 0
            i = i + 2
            m = m + 1
        
        j, m, n = 0,0,0
        i = 1
        while i < 2 * self.size:
            while j < self.size:
                errors_temp[1, m, n] = errors[i, j]
                j = j + 1
                n = n + 1
            n = 0
            j = 0
            i = i + 2
            m = m + 1
        
        count = [0, 0]
        for i in range(self.size):
            for j in range(self.size):
                if errors_temp[0, i, j] == 3 or errors_temp[0, i, j] == 2:
                    count[0] += 1
        
        for i in range(self.size):
            for j in range(self.size):
                if errors_temp[1, i, j] == 3 or errors_temp[1, i, j] == 2:
                    count[1] += 1
        #print (count)
        return count[0] % 2 == 0 and count[1] % 2 == 0

    def decode_X_error(self, errors, u, v):
        u = list(u)
        v = list(v)
        while u[1] != v[1]:
            if u[1] > v[1]:
                u, v = v, u
            x = u[0] * 2 + 1
            if v[1] - u[1] > self.size // 2:
                y = u[1] % self.size
                if errors[x, y] % 2 == 0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[1] -= 1
                u[1] %= self.size
            else:
                y = (u[1] + 1) % self.size
                if errors[x, y] % 2 == 0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[1] += 1
                u[1] %= self.size

        while u[0] != v[0]:
            if u[0] > v[0]:
                u, v = v, u
            y = u[1]
            if (v[0] - u[0]) > self.size // 2:
                x = 2 * u[0]
                if errors[x, y] % 2 == 0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[0] -= 1
                u[0] %= self.size
            else:
                x = (2 * u[0] + 2) % (2 * self.size)
                if errors[x, y] % 2 == 0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[0] += 1
                u[0] %= self.size
        return errors

    def decode_Z_error(self, errors, u, v):
        u = list(u)
        v = list(v)
        while u[1] != v[1]:
            if u[1] > v[1]:
                u, v = v, u
            x = 2 * u[0]
            if v[1] - u[1] > self.size // 2:
                y = (u[1] - 1) % self.size
                errors[x, y] = 3 - errors[x, y]
                u[1] -= 1
                u[1] %= self.size
            else:
                y = u[1] % self.size
                errors[x, y] = 3 - errors[x, y]
                u[1] += 1
                u[1] %= self.size
        while u[0] != v[0]:
            if u[0] > v[0]:
                u, v = v, u
            y = u[1]
            if v[0] - u[0] > self.size // 2:
                x = (2 * u[0] - 1) % (2 * self.size)
                errors[x, y] = 3 - errors[x, y]
                u[0] -= 1
                u[0] %= self.size
            else:
                x = (2 * u[0] + 1) % (2 * self.size)
                errors[x, y] = 3 - errors[x, y]
                u[0] += 1
                u[0] %= self.size
        return errors


# test
"""
param_test = toric_param.toric_param(p=0.10, size=5)
toric_code = ToricCode(param_test)
errors = toric_code.generate_errors()
errors_X = np.where(((errors==0) | (errors==3)), 0, 1)
errors_Z = np.where(((errors == 0)| (errors==1)), 0, 1)
print("errors:")
print(errors)
print(errors.shape)
errors_reshape_X = toric_code.errors_reshape(errors_X)
errors_reshape_Z = toric_code.errors_reshape(errors_Z)
print("reshapedX:", errors_reshape_X)
print("reshapedZ:", errors_reshape_Z)
print(errors_reshape_X.shape)
#print("tensor: ", errors_tensor)
#print("onehot:", errors_onehot)
#print("re-errors", re_errors)
#print(errors_onehot.size())
print("errors_X")
print(errors_X)
print("errors_Z")
print(errors_Z)
print("syndX")
syndrome_X = toric_code.generate_syndrome_X(errors)
print(syndrome_X)
print("syndX_diff")
syndrome_X_diff = toric_code.generate_syndrome_X_differentiable(errors_reshape_Z)
print(syndrome_X_diff)
print(syndrome_X_diff.shape)
print("syndZ")
syndrome_Z = toric_code.generate_syndrome_Z(errors)
print(syndrome_Z)
print("syndZ_diff")
syndrome_Z_diff = toric_code.generate_syndrome_Z_differentiable(errors_reshape_X)
print(syndrome_Z_diff)
print(syndrome_Z_diff.shape)
"""