""""
Demonstrate custom implementation of backward propagation of matrix multiplication
"""
import torch

# A is a (MxP) matrix and B is a (PxN) matrix, so C=AxB is a (MxN) matrix
M, P, N = 2, 3, 4
A = torch.randint(0, 100, (M, P), requires_grad=True, dtype=torch.float64)
B = torch.randint(0, 100, (P, N), requires_grad=True, dtype=torch.float64)

C = A.mm(B)
C.retain_grad()

# calculate the loss simply as the mean of C
loss = C.mean()

# perform build-in backpropagation
loss.backward()

print('A=', A)
print('B=', B)
print('C=', C)

print('built-in dL/dC=', C.grad)
print('built-in dL/dA=', A.grad)
print('built-in dL/dB=', B.grad)

# Now perform maunal calculation of the gradients dL/dC, dL/dA and dL/dB
grad_C_manual = (torch.ones(C.shape, dtype=torch.float64)/C.numel())
grad_A_manual = grad_C_manual.mm(B.t())
grad_B_manual = A.t().mm(grad_C_manual)

print('manual   dL/dC=', grad_C_manual)
print('manual   dL/dA=', grad_A_manual)
print('manual   dL/dA=', grad_B_manual)

diff_grad_C = grad_C_manual - C.grad
diff_grad_A = grad_A_manual - A.grad
diff_grad_B = grad_B_manual - B.grad

print('\nDifference between custom implementation and Torch built-in:')
print('diff_grad_C max difference:', diff_grad_C.abs().max().detach().numpy())
print('diff_grad_A max difference:', diff_grad_A.abs().max().detach().numpy())
print('diff_grad_B max difference:', diff_grad_B.abs().max().detach().numpy())

print('Done!')
