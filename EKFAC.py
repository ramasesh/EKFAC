import torch
import warnings

class EKFAC(torch.optim.Optimizer):
    """ Implements the Eigenvalue-corrected Kronecker-factored Optimized Curvature preconditioner

    See details at https://arxiv.org/pdf/1806.03884.pdf"""

    def __init__(self,
                 network,
                 recompute_KFAC_steps=10,
                 epsilon=0.1):
        """
        Arguments:
            network - the network to operate on
            recompute_KFAC_steps (integer) - the number of steps between successive recomputations of the
                                             Kronecker factors of the layer-wise Fisher matrix
            epsilon (float) - the damping parameter used to avoid infinities"""

        self.epsilon = epsilon

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
        self.precondition()

        self.iteration_number += 1

    def store_input(self, module, inputs_to_module):
        """ When called before running each layer with weights, this function stores
        the input to the layer"""

        self.stored_items[module]['input'] = inputs_to_module[0].t()

    def store_grad_output(self, module, grad_wrt_input, grad_wrt_output):
        """ When called after the backward pass of each layer with weights, this function
        stores the gradient of the backwards-running function (usually the loss function) with respect
        to the pre-activations, i.e. the output of the layer"""

        self.stored_items[module]['grad_wrt_output'] = grad_wrt_output[0]

    def compute_Kronecker_matrices(self):
        """ For each layer (or, more properly, parameter group), computes the Kronecker-factored matrices,
        where the Kronecker factors are defined by
        A = E[input_to_layer @ input_to_layer.T]
        B = E[grad_wrt_output @ grad_wrt_output.T]
        """

        for layer, stored_values in self.stored_items.items():
            # notation follows the EKFAC paper
            h = stored_values['input']
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
            h = stored_values['input']
            delta = stored_values['grad_wrt_output']

            with torch.no_grad():
                batch_size = h.shape[1]
                # TODO Check that this is correct
                # Because delta and h contain information for each training example in the mini-batch,
                # when we do the matrix multiplication in the middle, we are averaging over the mini-batch.
                # So, we need to square the values first, so we can square-then-average, not average-then-square.
                scalings = ((UB @ delta.t())**2) @ ((h.t() @ UA.t())**2) / batch_size

            stored_values['scalings'] = scalings

    def precondition(self):
        for layer, stored_values in self.stored_items.items():

            UA = stored_values['UA']
            UB = stored_values['UB']

            S = stored_values['scalings']

            grad_mb = layer.weight.grad.data # mb stands for 'mini-batch'
            grad_mb_kfe = UB @ grad_mb @ UA.t()
            grad_mb_kfe_scaled = grad_mb_kfe / (S + self.epsilon)
            grad_mb_orig = UB.t() @ grad_mb @ UA # back to original basis

            layer.weight.grad.data = grad_mb_orig

    def approximate_Fisher_matrix(self):
        """ For testing/debugging, compute the layer-wise approximation to the empirical Fisher matrix
            to compare to the Fischer information matrix """
        for layer, stored_values in self.stored_items.items():

            UA = stored_values['UA'].numpy()
            UB = stored_values['UB'].numpy()
            S = np.diag(stored_values['scalings'].numpy().reshape(-1))

            UAkronUB = np.kron(UA, UB)

            approximate_Fisher = UAkronUB @ S @ UAkronUB.T

            stored_values['aproximate_Fisher'] = torch.tensor(approximate_Fisher)

    def empirical_Fisher_matrix(self):
        """ For testing/debugging, compute the layer-wise empirical Fisher matrix"""
