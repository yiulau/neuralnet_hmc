import numpy

def gradient(pos,mom):
    return pos,mom
def leapfrog(pos, mom, step):
    # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
    #dE_dmom = grad_e(pos,mom)[1]
    #new_pos = pos + step * dE_dmom
    #dE_dpos = grad_e(new_pos,mom)[0]
    #new_pos = pos + step * grad_e(pos,mom)[1]
    #new_mom = mom - step * grad_e(new_pos,mom)[0]
    #new_mom = mom - step * dE_dpos
    # from vel(t+stepsize//2) compute pos(t+stepsize)
    new_pos = pos + step * mom
    new_mom = mom - step * new_pos
    return [new_pos, new_mom]

def simulate_dynamics(initial_pos, initial_mom, stepsize, n_steps):
    # define gradient function giving the partial derivatives
    # where first entry returns pd wrt position second entry wrt momentum
    grad_e = gradient
    # compute velocity at time-step: t + stepsize//2
    #dE_dpos = grad_e(initial_pos, initial_mom)[0]
    #mom_half_step = initial_mom - 0.5 * stepsize * dE_dpos
    mom_half_step = initial_mom - 0.5 * stepsize * initial_pos
    # perform leapfrog updates: the scan op is used to reatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    temp_pos = initial_pos
    temp_mom = mom_half_step
    for i in range(n_steps):
        temp_pos = temp_pos + stepsize * temp_mom
        if(i!=(n_steps-1)):
            #o = leapfrog(temp_pos,temp_mom,stepsize)
            #temp_pos = o[0]
            #temp_mom = o[1]
            temp_mom = temp_mom - stepsize * temp_pos

    final_mom = temp_mom - 0.5 * stepsize * temp_pos
    final_pos = temp_pos

    # return new proposal state
    return final_pos, final_mom
def simulate_dynamics_num(initial_pos, initial_mom, stepsize, n_steps):
    # define gradient function giving the partial derivatives
    # where first entry returns pd wrt position second entry wrt momentum
    grad_e = gradient
    def leapfrog(pos, mom, step):
        # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
        dE_dmom = grad_e(pos,mom)[1]
        new_pos = pos + step * dE_dmom
        dE_dpos = grad_e(new_pos,mom)[0]
        new_mom = mom - step * dE_dpos
        # from vel(t+stepsize//2) compute pos(t+stepsize)

        return [new_pos, new_mom]

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

# test energy function
def normal_en(pos,mom):
    return (numpy.dot(pos,pos) * 0.5 + numpy.dot(mom,mom) * 0.5)


initial_pos = 1.0
# fill this in later
energy_fn = lambda x: x + 1
#output = simulate_dynamics(initial_pos,initial_mom,stepsize,n_steps,energy_fn)
#print(output)
stepsize = 1.01
n_steps = 10
chain_length = 15000
store_array = numpy.zeros((chain_length,4))
store_array[0,1] = initial_pos
for j in range(1,chain_length):
    # simulate momentum
    fresh_mom = numpy.random.normal()
    out = simulate_dynamics(store_array[j-1,1],fresh_mom,stepsize,n_steps)
    proposed_pos = out[0]
    proposed_mom = out[1]
    #proposed_E = normal_en(proposed_pos,proposed_mom)
    #current_E = normal_en(store_array[i-1,1],fresh_mom)
    proposed_K = proposed_mom ** 2 * 0.5
    proposed_U = proposed_pos ** 2 * 0.5
    current_K = fresh_mom ** 2 * 0.5
    current_U = store_array[j-1,1] ** 2 * 0.5
    runif_v = numpy.random.uniform()
    accept = (runif_v < numpy.exp( current_U - proposed_U + current_K-proposed_K ))
    #accept = runif_v < numpy.exp(current_E - proposed_E))
    store_array[j,2]=proposed_pos
    store_array[j,3]=fresh_mom
    if accept:
        store_array[j,1] = proposed_pos
    else:
        store_array[j,1] = store_array[j-1,1]
    store_array[j,0] = accept

print(sum(store_array[:,0]))
print(numpy.mean(store_array[:,1]))
print(numpy.var(store_array[:,1]))
for i in range(10):
    print (store_array[i,1],store_array[i,2],store_array[i,3])