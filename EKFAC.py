import torch
import warnings

class EKFAC(torch.optim.Optimizer):
    """ Implements the Eigenvalue-corrected Kronecker-factored Optimized Curvature preconditioner

    See details at https://arxiv.org/pdf/1806.03884.pdf"""

    def __init__(self,
                 network,
                 recompute_KFAC_steps=1,
                 running_average=True,
                 alpha=0.75
                 epsilon=0.1):

        """
        Arguments:
            network - the network to operate on
            recompute_KFAC_steps (integer) - the number of steps between successive recomputations of the
                                             Kronecker factors of the layer-wise Fisher matrix
            epsilon (float) - the damping parameter used to avoid infinities
            running_average (bool) - if True, will keep a running average of the diagonal elements of the 
                                     scaling matrix (the approximate eigenvalues) rather than recomputing 
                                     from scratch for each iteration"""

        self.epsilon = epsilon
        self.running_average = running_average
        if self.running_average:
            self.alpha = alpha

        self.params_by_layer = []

        self.modules_with_weights = [torch.nn.Bilinear,
                        torch.nn.Conv1d,
                        torch.nn.Conv2d,
                        torch.nn.Conv3d,
                        torch.nn.ConvTranspose1d,
                        torch.nn.ConvTranspose2d,
                        torch.nn.ConvTranspose3d,
                        torch.nn.Linear,
                       ]

        self.stored_items = {}

        # need to keep track of iteration because we only recompute KFAC matrices every 'self.recompute_KFAC_steps' steps
        self.iteration_number = 0
        self.recompute_KFAC_steps = recompute_KFAC_steps

        tracked_modules_count = 0
        for layer in network.modules():
            if type(layer) in self.modules_with_weights:
                if type(layer) != torch.nn.Linear:
                    warnings.warn('Have not tested this for any module type other than linear')

                # add functions to the module such that for all layers with weights
                layer.register_forward_pre_hook(self.store_input)
                layer.register_backward_hook(self.store_grad_output)

                # add parameters to the list, grouped by layer
                self.params_by_layer.append({'params': [layer.weight]})
                if layer.bias is not None:
                    self.params_by_layer[-1]['params'].append(layer.bias)

                # make a label for the module and add it to the keys of the stored_items dictionary
                tracked_modules_count += 1

                self.stored_items[layer] = {}

        default_options = {}
        super(EKFAC, self).__init__(self.params_by_layer, default_options)

    def step(self):

        if self.iteration_number % self.recompute_KFAC_steps == 0:
            self.compute_Kronecker_matrices()

        self.compute_scalings()
#         self.precondition()

        self.iteration_number += 1

    def store_input(self, module, inputs_to_module):
        """ When called before running each layer with weights, this function stores
        the input to the layer"""

        inp = inputs_to_module[0]
        if module.bias is None:
            self.stored_items[module]['input'] = inp
        else:
            # if we have a bias, we use the trick of pretending that the input x is augmented with
            # a vector of ones
            self.stored_items[module]['input'] = torch.cat((inp, torch.ones((inp.size(0), 1))), dim=1)

    def store_grad_output(self, module, grad_wrt_input, grad_wrt_output):
        """ When called after the backward pass of each layer with weights, this function
        stores the gradient of the backwards-running function (usually the loss function) with respect
        to the pre-activations, i.e. the output of the layer"""

        """ We have to scale by the batch size, because the grad_wrt_output which is passed to the
         function is already scaled down by batch_size, even though we did not do any reduction """

        self.stored_items[module]['grad_wrt_output'] = grad_wrt_output[0] * grad_wrt_output[0].size(0)

    def compute_Kronecker_matrices(self):
        """ For each layer (or, more properly, parameter group), computes the Kronecker-factored matrices,
        where the Kronecker factors are defined by
        A = E[input_to_layer @ input_to_layer.T]
        B = E[grad_wrt_output @ grad_wrt_output.T]
        """

        for layer, stored_values in self.stored_items.items():
            # notation follows the EKFAC paper
            h = stored_values['input'].t()
            delta = stored_values['grad_wrt_output']

            # We want E[ h @ h.T]
            # h should always be of size (n_inputs, batch_size)
            # delta should be of size (batch_size, n_outputs)
            with torch.no_grad():
                A = h @ h.transpose(1,0) / h.shape[1]
                B = delta.transpose(1,0) @ delta / delta.shape[0]

            # Eigendecompose A and B to get UA and UB, which contain the eigenvectors
            # UA @ diag(EvalsA) @ UA.t() = A
            EvalsA, UA = torch.symeig(A, eigenvectors=True)
            EvalsB, UB = torch.symeig(B, eigenvectors=True)

            self.stored_items[layer]['UA'] = UA
            self.stored_items[layer]['UB'] = UB

    def compute_scalings(self):

        for layer, stored_values in self.stored_items.items():
            UA = stored_values['UA']
            UB = stored_values['UB']
            h = stored_values['input'].t()
            delta = stored_values['grad_wrt_output']

            with torch.no_grad():
                batch_size = h.shape[1]
                # Because delta and h contain information for each training example in the mini-batch,
                # when we do the matrix multiplication in the middle, we are averaging over the mini-batch.
                # So, we need to square the values first, so we can square-then-average, not average-then-square.
                scalings = ((UB.t() @ delta.t())**2) @ ((h.t() @ UA)**2) / batch_size

            if not self.running_average:
                stored_values['scalings'] = scalings
            else:
                stored_values['scalings'] = self.alpha * scalings + (1 - self.alpha) * stored_values['scalings']

    def precondition(self):
        for layer, stored_values in self.stored_items.items():

            UA = stored_values['UA']
            UB = stored_values['UB']

            S = stored_values['scalings']

            if layer.bias is None:
                grad_mb = layer.weight.grad # mb stands for 'mini-batch'
                grad_mb_kfe = UB @ grad_mb @ UA.t()
                grad_mb_kfe_scaled = grad_mb_kfe / (S + self.epsilon)
                grad_mb_orig = UB.t() @ grad_mb @ UA # back to original basis
                layer.weight.grad.data = grad_mb_orig
            else:
                grad_mb_W = layer.weight.grad
                grad_mb_b = layer.bias.grad
                grad_mb = torch.cat(grad_mb_W, grad_mb_b, dim=1)
                grad_mb_kfe = UB @ grad_mb @ UA.t()
                grad_mb_kfe_scaled = grad_mb_kfe / (S + self.epsilon)
                grad_mb_orig = UB.t() @ grad_mb @ UA # back to original basis
    
                layer.weight.grad.data = grad_mb_orig[:,:-1]
                layer.bias.grad.data = grad_mb_orig[:,-1]

    def approximate_Fisher_matrix(self, to_return=False):
        """ For testing/debugging, compute the layer-wise approximation to the empirical Fisher matrix
            to compare to the Fischer information matrix """
        approximate_Fisher_matrices = []
        for layer, stored_values in self.stored_items.items():

            UA = stored_values['UA'].numpy()
            UB = stored_values['UB'].numpy()
            S = np.diag(stored_values['scalings'].numpy().reshape(-1))

            UAkronUB = np.kron(UA, UB)

            approximate_Fisher = UAkronUB @ S @ UAkronUB.T
            approximate_Fisher_matrices.append(approximate_Fisher)

            stored_values['aproximate_Fisher'] = torch.tensor(approximate_Fisher)

        if to_return:
            return approximate_Fisher_matrices

    def compute_hdeltaT(self):
        """ For testing/debugging, compute the layer-wise h delta T product.
        The minibatch-averaged h delta^T product should be equal to the gradient of the
        weigt matrix for each linear layer."""

        for layer, stored_values in self.stored_items.items():
            h = stored_values['input']
            delta = stored_values['grad_wrt_output']
            stored_values['hdeltaT'] = h.t() @ delta / h.size(0)

    def compute_empirical_Fisher_matrix(self, to_return=False):
        """ For testing/debugging, compute empirical Fisher matrix """

        empirical_fisher_matrices = []
        for layer, stored_values in self.stored_items.items():
            h = stored_values['input']
            delta = stored_values['grad_wrt_output']

            with torch.no_grad():
                empirical_fisher_matrix = empirical_fisher(h, delta)
                stored_values['empirical_fisher'] = empirical_fisher_matrix
                empirical_fisher_matrices.append(empirical_fisher_matrix)

        if to_return:
            return empirical_fisher_matrices

def outer_prod_individual(M1, M2):
    """ takes the outer product of M1 and M2, where M1 is NxA, and M2 is NxB
    """
    return torch.einsum('ij,ik->ijk', M1, M2)

def vectorize_individual(M):
    """ Given a tensor M, with size (A, B, C), vectorizes this tensor, leaving the first dimension intact,
    by stacking columns, resulting in a tensor of size (A, BC)"""
    Mt = M.transpose(1,2)
    return Mt.contiguous().view(Mt.size(0), -1)

def empirical_fisher(h, delta):
    """ given h, representing the input to a layer, and delta, representing the gradient with respect to its output,
    computes the empirical fisher matrix, averaged over the minibatch

    Arguments:
    h - torch.tensor, dimension (batch_size) * (n_inputs)
    delta - torch.tensor, dimension (batch_size) * (n_outputs)"""

    grad_individual = outer_prod_individual(h, delta)
    vec_grad_individual = vectorize_individual(grad_individual)
    fisher_individual = outer_prod_individual(vec_grad_individual, vec_grad_individual)
    return fisher_individual.mean(0)
