# Check that the trajectories returned are the same
import theano.tensor as T
import numpy, theano
step = 1.2
beta_0 = T.fvector()
p_0 = T.fvector()
energy_fn = lambda beta_0,p_0 : T.dot(beta_0,beta_0)*0.5 + T.dot(p_0,p_0)*0.5
#
def simulate_dynamics(initial_pos, initial_mom, stepsize, n_steps, energy_fn):

    def leapfrog(pos, mom, step):
        # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
        dE_dmom = T.grad(energy_fn(pos,mom),mom)
        new_pos = pos + step * dE_dmom
        dE_dpos = T.grad(energy_fn(new_pos,mom),new_pos)
        new_mom = mom - step * dE_dpos
        # from vel(t+stepsize//2) compute pos(t+stepsize)

        return [new_pos, new_mom]

    # compute velocity at time-step: t + stepsize//2
    initial_energy = energy_fn(initial_pos,initial_mom)
    dE_dpos = T.grad(initial_energy, initial_pos)
    mom_half_step = initial_mom - 0.5 * stepsize * dE_dpos


    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    (all_pos, all_mom), scan_updates = theano.scan(
        leapfrog,
        outputs_info=[
            dict(initial=initial_pos),
            dict(initial=mom_half_step),
        ],
        non_sequences=[stepsize],
        n_steps=n_steps -1)
    final_pos = all_pos[-1]
    final_mom = all_mom[-1]
    # NOTE: Scan always returns an updates dictionary, in case the
    # scanned function draws samples from a RandomStream. These
    # updates must then be used when compiling the Theano function, to
    # avoid drawing the same random numbers each time the function is
    # called. In this case however, we consciously ignore
    # "scan_updates" because we know it is empty.
    assert not scan_updates

    # The last velocity returned by scan is vel(t +
    # (n_steps - 1 / 2) * stepsize) We therefore perform one more half-step
    # to return vel(t + n_steps * stepsize)
    final_energy = energy_fn(final_pos,final_mom)
    output_mom = final_mom - 0.5 * stepsize * T.grad(final_energy, final_pos)

    return final_pos,output_mom
    # return new proposal state
    #return final_pos, output_mom,scan_updates,all_pos,all_mom

def gradient(pos,mom):
    return pos,mom


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
    for i in range(1,n_steps):
        o = leapfrog(temp_pos,temp_mom,stepsize)
        temp_pos = o[0]
        temp_mom = o[1]

    final_mom = temp_mom - 0.5 * stepsize * grad_e(temp_pos,temp_mom)[0]
    final_pos = temp_pos

    # return new proposal state
    return final_pos, final_mom

num_steps=10
init_pos = numpy.asarray([1.0],dtype=theano.config.floatX)
init_mom = numpy.asarray([-1.5],dtype=theano.config.floatX)
out = simulate_dynamics(beta_0,p_0,step,num_steps,energy_fn)
#f_theano = theano.function([beta_0,p_0],out[3:5],updates=out[2])
f_theano = theano.function([beta_0,p_0],out)
real_output = f_theano(init_pos,init_mom)
print("Output by theano is {}".format(real_output))
#print("Theano trajectory is {}".format(real_output[1]))
out_num = simulate_dynamics_num(init_pos,init_mom,step,num_steps)
print("Output by numpy is {}".format(out_num))