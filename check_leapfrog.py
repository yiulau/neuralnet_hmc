# Check that the two leapfrogs return the same value
import theano.tensor as T
import numpy, theano
step = 1.9
beta_0 = T.fvector()
p_0 = T.fvector()
energy_fn = lambda beta_0,p_0 : T.dot(beta_0,beta_0)*0.5 + T.dot(p_0,p_0)*0.5
#
def leapfrog(pos, mom):
    # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
    dE_dmom = T.grad(energy_fn(pos, mom), mom)
    new_pos = pos + step * dE_dmom
    dE_dpos = T.grad(energy_fn(new_pos, mom), pos)
    new_mom = mom - step * dE_dpos
    # from vel(t+stepsize//2) compute pos(t+stepsize)

    return [new_pos, new_mom]
def gradient(pos,mom):
    return pos,mom

grad_e = gradient
def leapfrog_num(pos, mom, step):
    # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
    dE_dmom = grad_e(pos,mom)[1]
    new_pos = pos + step * dE_dmom
    dE_dpos = grad_e(new_pos,mom)[0]
    new_mom = mom - step * dE_dpos
    # from vel(t+stepsize//2) compute pos(t+stepsize)

    return [new_pos, new_mom]

# test
init_mom = numpy.asarray([1.0],theano.config.floatX)
init_pos = numpy.asarray([-1.5],dtype=theano.config.floatX)

out = leapfrog(beta_0,p_0)
f_theano = theano.function([beta_0,p_0],out)

theano_out = f_theano(init_pos,init_mom)
numpy_out = leapfrog_num(init_pos,init_mom,step)

print("Output by theano is {}".format(theano_out))
print("Output by numpy is {}".format(numpy_out))

