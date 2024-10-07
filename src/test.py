import torch
import torch.nn as nn

from time import time


bs = 2
n_passage = 100
max_langth = 200
dimension = 768

Linear_a = nn.Linear(dimension, dimension)
Linear_p = nn.Linear(dimension, dimension)


t = torch.rand([bs, n_passage, max_langth, dimension])

# def quantum_fusion(t):
#     # 量子融合操作
#     bs, n_passage, max_langth, dimension = t.shape

#     # [400, 200, 768]    
#     t = t.reshape(bs * n_passage, max_langth, dimension)

#     rets = []
#     for i in range(t.shape[0]):
#         # 依次进行fusion
#         # 维度为 [200, 768]
#         psi_real = t[i]
#         psi_imag = Linear(psi_real)
#         amplitude = torch.sqrt(psi_real**2 + psi_imag**2)
#         phase = torch.atan2(psi_imag, psi_real)
#         psi = amplitude * torch.exp(1j * phase)
#         average_psi = torch.sum(psi, dim=0)
#         r = torch.real(average_psi)

#         rets.append(r)

#     rets = torch.stack(rets)
#     return rets.reshape(bs, n_passage, dimension)


def max_state(t):
    bs, n_passage, max_langth, dimension = t.shape
    t = t.reshape(bs * n_passage, max_langth, dimension)
    max_values, _ = torch.max(t, dim=1)
    return max_values.reshape(bs, n_passage, dimension)


def quantum_fusion_amplitude(t):
    now = time()
    rets = []
    bs, n_passage, max_length, dimension = t.shape
    rs_t = t.reshape(bs * n_passage, max_length, dimension)  # [200, 200, 768]
    amplitudes = Linear_a(rs_t) # [200, 200, 768]
    phases = Linear_p(rs_t) # [200, 200, 768]
    print(time() - now)
    now = time()
    for psi, amplitude, phase in zip(rs_t, amplitudes, phases):
        p = amplitude * torch.exp(1j * phase)
        lo = torch.matmul(p, p.T)
        eigenvalues = torch.diag(lo)
        p_eigen = eigenvalues / sum(eigenvalues)
        sum_psi = torch.sum(p_eigen[:, None] * psi, dim=0)
    
        r = torch.real(sum_psi)
        rets.append(r)
    
    print(time() - now)
    now = time()
    p = amplitude * torch.exp(1j * phase) # [200, 200, 768]
    pT = p.transpose(1,2) # [200, 768, 200]
    
    lo = torch.matmul(p, pT)
    # print(time() - now)
    # now = time()
    for psi, l in zip(rs_t, lo):
        eigenvalues = torch.diag(l)
        p_eigen = eigenvalues / sum(eigenvalues)
        sum_psi = torch.sum(p_eigen[:, None] * psi, dim=0)
        r = torch.real(sum_psi)
        rets.append(r)
    rets = torch.stack(rets, dim=0)

    print(time() - now)
    return sum_psi



t = quantum_fusion_amplitude(t)


# print(t.shape)