import numpy
import torch
from torch.autograd import Variable

dim = 2
chain_l = 4000
q = torch.rand(dim)
q = Variable(q,requires_grad=True)

p = torch.rand(dim)
p = Variable(p,requires_grad=True)
Sig = Variable(torch.diag(torch.ones(dim)))

potential = 0.5 * torch.dot(q,torch.mv(Sig,q))
potential.backward(retain_graph=True)


kinetic = torch.dot(p,p) * 0.5

energy = kinetic + potential
store = torch.rand((chain_l,dim))

def HMC(epsilon,L,current_q):
    p.data.normal_()
    q.data = current_q.data
    current_K = torch.dot(p,p) * 0.5
    current_U = 0.5 * torch.dot(q,torch.mv(Sig,q))
    potential = current_U.clone()
    potential.backward(retain_graph=True)
    p.data = p.data - epsilon * q.grad.data * 0.5
    q.grad.data.zero_()
    for i in range(L-1):
        q.data = q.data + epsilon * p.data
        potential = 0.5 * torch.dot(q,torch.mv(Sig,q))
        potential.backward(retain_graph=True)
        p.data = p.data - epsilon * q.grad.data
        q.grad.data.zero_()
    potential = 0.5 * torch.dot(q,torch.mv(Sig,q))
    potential.backward(retain_graph=True)
    p.data = p.data - epsilon * q.grad.data * 0.5
    p.data = -p.data
    proposed_U = 0.5 * torch.dot(q,torch.mv(Sig,q))
    proposed_K = torch.dot(p,p) * 0.5
    proposed_U = proposed_U.data
    proposed_K = proposed_K.data
    current_K = current_K.data
    current_U = current_U.data
    temp = torch.exp(current_U - proposed_U + current_K - proposed_K)

    print("current U is {}".format(current_U))
    print("proposed U is {}".format(proposed_U))
    print("current K is {}".format(current_K))
    print("propsed K is {}".format(proposed_K))
    print("temp is {}".format(temp))
    if(numpy.random.random(1) < temp.numpy()):
        return(q.data)
    else:
        return(current_q.data)


for i in range(chain_l):
    print("round {}".format(i))
    out = HMC(0.1,10,q)
    store[i,]=out
    q.data = out


#o=numpy.cov(store)
#print(o)
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
print(empCov)
emmean = numpy.mean(store,axis=0)
print(emmean)
#print(store[1:20,4])
