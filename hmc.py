# Bayesian logistic regression flat prior

import theano,numpy,math
import theano.tensor as T
#from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
X = T.fmatrix()
y = T.fmatrix()
beta = theano.shared(numpy.asarray(numpy.random.randn(784,1), dtype=theano.config.floatX))
py_x = T.nnet.softmax(T.dot(X,beta))
y_pred = T.argmax(beta,axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x,y))

# energy function for normal distribution with normal momentum
def normal_en(pos,mom):
    total_en = T.dot(pos,pos)/2 + T.dot(mom,mom)/2
    f = theano.function([pos,mom],total_en)
    return(f)

beta_0 = T.fvector()
p_0 = T.fvector()
en = lambda beta_0,p_0 : T.dot(beta_0,beta_0)*0.5 + T.dot(p_0,p_0)*0.5
#en_f = theano.function([],en)

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
        n_steps=n_steps - 1)
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
    energy = energy_fn(final_pos,final_mom)
    final_mom = final_mom - 0.5 * stepsize * T.grad(energy, final_pos)

    # return new proposal state
    return final_pos, final_mom

def metropolis_hastings_accept(energy_prev,energy_next,s_rng):
    ediff = energy_prev - energy_next
    ran = s_rng.uniform([1])
    t = T.exp(ediff) > ran
    return t,ran

# start-snippet-1
def hmc_move(s_rng, initial_pos, energy_fn, stepsize, n_steps):
    # end-snippet-1 start-snippet-2
    # sample random velocity
    initial_mom = s_rng.normal(size=initial_pos.shape)
    # end-snippet-2 start-snippet
    # perform simulation of particles subject to Hamiltonian dynamics
    final_pos, final_mom = simulate_dynamics(
        initial_pos=initial_pos,
        initial_mom=initial_mom,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    # end-snippet-3 start-snippet-4
    # accept/reject the proposed move based on the joint distribution

    accept = metropolis_hastings_accept(
        energy_prev=energy_fn(initial_pos, initial_mom),
        energy_next=energy_fn(final_pos, final_mom),
        s_rng=s_rng
    )
    #next_pos = ifelse(accept,initial_pos,initial_pos)
    # end-snippet-4
    #print(ifelse(accept,1,2))
    #return accept,next_pos,final_pos,initial_pos
    return accept[0],final_pos,initial_pos,final_mom,initial_mom,accept[1]
#out = en_f(beta_0,p_0)
rng = RandomStreams()
out = hmc_move(rng,beta_0,en,0.1,10)
testf = theano.function([beta_0],out)

# Try to sample from normal disribution
initialvalue = 0.15
samplesize = 10000
storearray = numpy.asarray(numpy.zeros((samplesize,2)),dtype=theano.config.floatX)
o = testf(numpy.asarray([initialvalue], dtype=theano.config.floatX))
print(o)
en_next = (o[1]**2 + o[3]**2)*0.5
en_prev = (o[2]**2 + o[4]**2)*0.5

exit()
storearray[0,0]=o[0]
if o[0]:
    storearray[0,1]=o[1]
else:
    storearray[0,1]=o[2]
print(storearray[0,])

for i in range(samplesize-1):
    initial = storearray[i,1]
    o = testf(numpy.asarray([initial],dtype=theano.config.floatX))
    storearray[i,0]=o[0]
    if o[0]:
        storearray[i,1]=o[1]
    else:
        storearray[i,1]=o[2]
print(numpy.mean(storearray[:,1]))
print(numpy.var(storearray[:,1]))
print(numpy.mean(storearray[:,0]))

"""""
def leapfrog(pos, mom, step,energy_fn):
    # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
    dE_dmom = T.grad(energy_fn(pos, mom), mom)
    new_pos = pos + step * dE_dmom
    dE_dpos = T.grad(energy_fn(new_pos, mom), new_pos)
    new_mom = mom - step * dE_dpos
    # from vel(t+stepsize//2) compute pos(t+stepsize)

    return [new_pos,new_mom]

#out = leapfrog(beta_0, p_0, 0.1, en)
out = simulate_dynamics(beta_0,p_0,0.1,10,en)
import time
start = time.time()
test_f = theano.function([beta_0,p_0],out)
time1 = time.time()-start
start = time.time()
for i in range(100):
    o = test_f([0],[1])
    print(o)
time2 = time.time()-start
print("Time to compile is {}".format(time1))
print("Time to run is {}".format(time2))
"""