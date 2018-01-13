import numpy
import torch
from torch.autograd import Variable

dim = 10
chain_l = 100
q = torch.rand(dim)
q = Variable(q,requires_grad=True)

p = torch.rand(dim)
p = Variable(p,requires_grad=True)
Sig = Variable(torch.ones((dim,dim)))

potential = 0.5 * torch.dot(q,torch.mv(Sig,q))
kinetic = torch.dot(p,p) * 0.5
energy = kinetic + potential
store = torch.rand((chain_l,dim))

def HMC(epsilon,L,current_q):
    p.data.normal_()
    q.data = current_q.data.clone()
    current_K = kinetic.data.clone()
    current_U = potential.data.clone()
    potential.backward(retain_graph=True)
    p.data = p.data - epsilon * q.grad.data * 0.5
    q.grad.data.zero_()
    for i in range(L-1):
        q.data = q.data + epsilon * p.data
        potential.backward(retain_graph=True)
        p.data = p.data - epsilon * q.grad.data
        q.grad.data.zero_()
    potential.backward(retain_graph=True)
    p.data = p.data - epsilon * q.grad.data * 0.5
    p.data = -p.data
    proposed_U = 0.5 * torch.dot(q,torch.mv(Sig,q))
    proposed_K = torch.dot(p,p) * 0.5
    proposed_U = proposed_U.data
    proposed_K = proposed_K.data
    temp = torch.exp(current_U - proposed_U + current_K - proposed_K)

    #print(current_U)
    #print(proposed_U)
    #print(current_K)
    #print(proposed_K)
    #print(temp)
    if(numpy.random.random(1) < temp.numpy()):
        return(q.data)
    else:
        return(current_q.data)


for i in range(chain_l):

    out = HMC(0.1,20,q)
    store[i,]=out
    q.data = out

print(store[1:70,0])
