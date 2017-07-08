import torch
from torch.autograd import Function
from torch.nn import Module
import numpy as np

def project_onto_simplex(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.copy()
    d = len(x0);
    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]
    ycum = np.cumsum(y0)
    val = 1.0/np.arange(1,d+1) * (ycum - radius)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]
    y = y0 - tau
    ind = np.nonzero(y < 0)
    y[ind] = 0
    x = x0.copy()
    x[ind_sort] = y
    return x, tau, .5*np.dot(x-a, x-a)

class SoftmaxFunction(Function):
    def forward(self, input):
        e_z = input.exp()
        Z = e_z.sum(1)
        output = e_z / Z.expand_as(e_z)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        avg = (grad_output * output).sum(1)
        grad_input = output * (grad_output - avg.expand_as(grad_output))
        return grad_input

class Softmax(Module):
    def forward(self, input):
        return SoftmaxFunction()(input)

class SparsemaxFunction(Function):
    def forward(self, input):
        # TODO: Make an implementation directly with torch tensors,
        # not requiring numpy.
        # Example:
        # z_sorted, ind_sort = (-input).sort(dim=1, descending=True)
        # z_cum = z_sorted.cumsum(dim=1)
        # r = torch.arange(1, 1+z_sorted.size(1))
        # if input.is_cuda():
        #     r = r.cuda()
        # val = 1.0 / r.expand_as(z_cum) * (z_cum - 1.)
        # ...
        np_input = input.cpu().numpy()
        probs = np.zeros_like(np_input)
        for i in xrange(np_input.shape[0]):
            probs[i,:], tau, _ = project_onto_simplex(np_input[i,:])
        output = torch.from_numpy(probs)
        if input.is_cuda:
            output = output.cuda()
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        probs = output.cpu().numpy()
        supp = np.array(probs > 0., dtype=probs.dtype)
        np_grad_output = grad_output.cpu().numpy()
        avg = np.sum(np_grad_output * supp, 1) / np.sum(supp, 1)
        np_grad_input = supp * (np_grad_output - np.tile(avg[:,None],
                                                         [1, supp.shape[1]]))
        grad_input = torch.from_numpy(np_grad_input)
        if grad_output.is_cuda:
           grad_input  = grad_input.cuda()
        return grad_input

class Sparsemax(Module):
    def forward(self, input):
        return SparsemaxFunction()(input)

class ConstrainedSoftmaxFunction(Function):
    def forward(self, input1, input2):
        z = input1
        u = input2
        e_z = z.exp()
        Z = e_z.sum(1)
        probs = e_z / Z.expand_as(e_z)
        active = (probs > u).type(probs.type())
        s = (active * u).sum(1)
        Z = ((1. - active) * e_z).sum(1) / (1-s)
        probs = active * u + (1. - active) * (e_z / Z.expand_as(z))
        output = probs
        self.save_for_backward(output)
        self.saved_intermediate = active, s # Not sure this is safe.
        return output

    def backward(self, grad_output):
        output, = self.saved_tensors
        active, s = self.saved_intermediate
        probs = output
        m = ((1. - active) * probs * grad_output).sum(1) / (1. - s)
        grad_z = (1. - active) * probs * (grad_output - m.expand_as(active))
        grad_u = active * (grad_output - m.expand_as(active))
        grad_input1 = grad_z
        grad_input2 = grad_u
        return grad_input1, grad_input2

class ConstrainedSoftmax(Module):
    def forward(self, input1, input2):
        return ConstrainedSoftmaxFunction()(input1, input2)


