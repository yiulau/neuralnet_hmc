import numpy

def grad(fn):
    def gradient(pos,mom):
        return pos,mom
    return gradient
def simulate_dynamics(initial_pos, initial_mom, stepsize, n_steps, energy_fn):
    # define gradient function giving the partial derivatives
    # where first entry returns pd wrt position second entry wrt momentum
    grad_e = grad(energy_fn)
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


initial_pos = 0.15
# fill this in later
initial_mom = 0.1
stepsize = 0.1
n_steps = 10
energy_fn = lambda x: x+1
output = simulate_dynamics(initial_pos,initial_mom,stepsize,n_steps,energy_fn)
print(output)
