import numpy

def gradient(pos,mom):
    return pos,mom
def simulate_dynamics(initial_pos, initial_mom, stepsize, n_steps):
    p = initial_mom - 0.5 * stepsize * initial_pos
    q = initial_pos
    for i in range(n_steps):
        q = q + stepsize * p
        if(i != (n_steps-1)):
            p = p - stepsize * q
    p = p - 0.5 * stepsize * q
    return (q, p)

def simulate_dynamics_num(initial_pos, initial_mom, stepsize, n_steps):
    # define gradient function giving the partial derivatives
    # where first entry returns pd wrt position second entry wrt momentum
    grad_e = gradient
    """""
    def leapfrog(pos, mom, step):
        # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
        dE_dmom = grad_e(pos,mom)[1]
        new_pos = pos + step * dE_dmom
        dE_dpos = grad_e(new_pos,mom)[0]
        new_mom = mom - step * dE_dpos
        # from vel(t+stepsize//2) compute pos(t+stepsize)

        return [new_pos, new_mom]
    """
    # compute velocity at time-step: t + stepsize//2
    dE_dpos = grad_e(initial_pos, initial_mom)[0]
    mom_half_step = initial_mom - 0.5 * stepsize * dE_dpos


    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    temp_pos = initial_pos
    temp_mom = mom_half_step
    for i in range(2,n_steps):
        o = leapfrog(temp_pos,temp_mom,stepsize)
        temp_pos = o[0]
        temp_mom = o[1]

    final_mom = temp_mom - 0.5 * stepsize * grad_e(temp_pos,temp_mom)[0]
    final_pos = temp_pos

    # return new proposal state
    return final_pos, final_mom
def HMC(U, gradU, epsilon, L, currentq,random_norm,random_unif):

    q = currentq
    p =  random_norm
    currentp = p
    p = p - epsilon * gradU(q) * 0.5

    for i in range(L):
        q = q + epsilon * p
        if (i != (L-1)):
            p = p - epsilon * gradU(q)


    p = p - epsilon * gradU(q) * 0.5
    p = -p
    #currentp = random_norm
    out = simulate_dynamics(currentq,random_norm,epsilon,L)
    q1 = out[0]
    p1 = out[1]
    print("Right q {}, wrong q{}".format(q,q1))
    print("Right p {},wrong p{}".format(p,p1))
    curU = U(currentq)
    curK = currentp ** 2 * 0.5
    proU = U(q)
    proK = p ** 2 * 0.5

    if (random_unif < numpy.exp(curU - proU + curK - proK)):
        return (q)
    else:
        return (currentq)


# negative log density
def target_f(x):

    return (x ** 2 * 0.5)

# gradient of negative log
def target_grad(x):

    return (x)

# Start sampling
num_draws = 2
epsilon = 0.98
L = 10
initq = 1
storematrix = numpy.zeros((num_draws, 2))
storematrix[0, 1] = initq

for i in range(1,num_draws):
    out = HMC(target_f, target_grad, epsilon, L,storematrix[i - 1, 1],numpy.random.normal(),numpy.random.uniform())
    if (out != storematrix[i - 1, 1]):
        storematrix[i, 0]=1
        storematrix[i,1] = out
    else:
        storematrix[i, 1] = storematrix[i-1,1]


acceptance_rate = sum(storematrix[:, 0]) / num_draws
#plot(storematrix[, 1])
print(acceptance_rate)
print(numpy.mean(storematrix[:, 1]))
print(numpy.var(storematrix[:, 1]))

