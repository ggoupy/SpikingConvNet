import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from torch.nn.functional import conv2d, max_pool2d

from utils import load_encoded_MNIST


"""
Implementation of the paper STDP-based spiking deep neural networks for object recognition
for the MNIST classification task.

References:
       
       [1] Kheradpisheh, S. R., Ganjtabesh, M., Thorpe, S. J., &#38; Masquelier, T. (2018). 
           STDP-based spiking deep convolutional neural networks for object recognition.
           Neural Networks, 99, 56–67. https://doi.org/10.1016/J.NEUNET.2017.12.005
  
       [2] Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., &#38; Masquelier, T. (2019).
           SpykeTorch: Efficient simulation of convolutional spiking neural networks with
           at most one spike per neuron.
           Frontiers in Neuroscience, 13, 625. https://doi.org/10.3389/FNINS.2019.00625
  
       [3] https://github.com/npvoid/SDNN_python
"""




class SpikingPool:
    """ 
    Pooling layer with spiking neurons that can fire only once.
    """
    def __init__(self, input_shape, kernel_size, stride, padding=0):
        in_channels, in_height, in_width = input_shape
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_shape = (in_channels, out_height, out_width)
        # Keep track of active neurons because they can fire once
        self.active_neurons = np.ones(self.output_shape).astype(bool)


    def reset(self):
        self.active_neurons[:] = True


    def __call__(self, in_spks):
        # padding 
        in_spks = np.pad(in_spks, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        in_spks = torch.Tensor(in_spks).unsqueeze(0)
        # Max pooling (using torch as it is fast and easier, to be changed)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride).numpy()[0]
        # Keep spikes of active neurons
        out_spks = out_spks * self.active_neurons
        # Update active neurons as each pooling neuron can fire only once
        self.active_neurons[out_spks == 1] = False
        return out_spks




class SpikingConv:
    """ 
    Convolutional layer with IF spiking neurons that can fire only once.
    Implements a Winner-take-all STDP learning rule.
    """
    def __init__(self, input_shape, out_channels, kernel_size, stride, padding=0, 
                nb_winners=1, firing_threshold=1, stdp_max_iter=None, adaptive_lr=False,
                stdp_a_plus=0.004, stdp_a_minus=-0.003, stdp_a_max=0.15, inhibition_radius=0,
                update_lr_cnt=500, weight_init_mean=0.8, weight_init_std=0.05, v_reset=0
        ):
        in_channels, in_height, in_width = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        self.firing_threshold = firing_threshold
        self.v_reset = v_reset
        self.weights = np.random.normal(
            loc=weight_init_mean, scale=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))

        # Output neurons
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.pot = np.zeros((out_channels, out_height, out_width))
        self.active_neurons = np.ones(self.pot.shape).astype(bool)
        self.output_shape = self.pot.shape

        # STDP
        self.recorded_spks = np.zeros((in_channels, in_height+2*self.padding[0], in_width+2*self.padding[1]))
        self.nb_winners = nb_winners
        self.inhibition_radius = inhibition_radius
        self.adaptive_lr = adaptive_lr
        self.a_plus = stdp_a_plus
        self.a_minus = stdp_a_minus
        self.a_max = stdp_a_max
        self.stdp_cnt = 0
        self.update_lr_cnt = update_lr_cnt
        self.stdp_max_iter = stdp_max_iter
        self.plasticity = True
        self.stdp_neurons = np.ones(self.pot.shape).astype(bool)


    def get_learning_convergence(self):
        return (self.weights * (1-self.weights)).sum() / np.prod(self.weights.shape)


    def reset(self):
        self.pot[:] = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:] = True
        self.recorded_spks[:] = 0


    def get_winners(self):
        winners = []
        channels = np.arange(self.pot.shape[0])
        # Copy potentials and keep neurons that can do STDP
        pots_tmp = np.copy(self.pot) * self.stdp_neurons
        # Find at most nb_winners
        while len(winners) < self.nb_winners:
            # Find new winner
            winner = np.argmax(pots_tmp) # 1D index
            winner = np.unravel_index(winner, pots_tmp.shape) # 3D index
            # Assert winner potential is higher than firing threshold
            # If not, stop the winner selection 
            if pots_tmp[winner] <= self.firing_threshold:
                break
            # Add winner
            winners.append(winner)
            # Disable winner selection for neurons in neighborhood of other channels
            pots_tmp[channels != winner[0],
                max(0,winner[1]-self.inhibition_radius):winner[1]+self.inhibition_radius+1,
                max(0,winner[2]-self.inhibition_radius):winner[2]+self.inhibition_radius+1
            ] = self.v_reset
            # Disable winner selection for neurons in same channel
            pots_tmp[winner[0]] = self.v_reset 
        return winners


    def lateral_inhibition(self, spks):
        # Get index of spikes
        spks_c,spks_h,spks_w = np.where(spks)
        # Get associated potentials
        spks_pot = np.array([self.pot[spks_c[i],spks_h[i],spks_w[i]] for i in range(len(spks_c))])
        # Sort index by potential in a descending order
        spks_sorted_ind = np.argsort(spks_pot)[::-1]
        # Sequentially inhibit neurons in the neighborhood of other channels
        # Neurons with highest potential inhibit neurons with lowest one, even if both spike
        for ind in spks_sorted_ind:
            # Check that neuron has not been inhibated by another one
            if spks[spks_c[ind],spks_h[ind],spks_w[ind]] == 1:
                # Compute index
                inhib_channels = np.arange(spks.shape[0]) != spks_c[ind]
                # Inhibit neurons
                spks[inhib_channels,spks_h[ind],spks_w[ind]] = 0 
                self.pot[inhib_channels,spks_h[ind],spks_w[ind]] = self.v_reset
                self.active_neurons[inhib_channels,spks_h[ind],spks_w[ind]] = False
        return spks


    def get_conv_of(self, input, output_neuron):
        # Neuron index
        n_c, n_h, n_w = output_neuron
        # Get the list of convolutions on input neurons to update output neurons
        # shape : (in_neuron_values, nb_convs)
        input = torch.Tensor(input).unsqueeze(0) # batch axis
        convs = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride)[0].numpy()
        # Get the convolution for the spiking neuron
        conv_ind = (n_h * self.pot.shape[2]) + n_w # 2D to 1D index
        return convs[:, conv_ind]

        
    def stdp(self, winner):
        if not self.stdp_neurons[winner]: exit(1)
        if not self.plasticity: return
        # Count call
        self.stdp_cnt += 1
        # Winner 3D coordinates
        winner_c, winner_h, winner_w = winner
        # Get convolution window used to compute output neuron potential
        conv = self.get_conv_of(self.recorded_spks, winner).flatten()
        # Compute dW
        w = self.weights[winner_c].flatten() * (1 - self.weights[winner_c]).flatten()
        w_plus = conv > 0 # Pre-then-post
        w_minus = conv == 0 # Post-then-pre (we assume that if no spike before, then after)
        dW = (w_plus * w * self.a_plus) + (w_minus * w * self.a_minus)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)
        # Lateral inhibition between channels (local inter competition)
        channels = np.arange(self.pot.shape[0])
        self.stdp_neurons[channels != winner_c,
            max(0,winner_h-self.inhibition_radius):winner_h+self.inhibition_radius+1,
            max(0,winner_w-self.inhibition_radius):winner_w+self.inhibition_radius+1
        ] = False
        # Lateral inhibition in the same channel (gobal intra competition)
        self.stdp_neurons[winner_c] = False
        # Adpative learning rate
        if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
            self.a_plus = min(2 * self.a_plus, self.a_max)
            self.a_minus = - 0.75 * self.a_plus
        # Stop STDP after X trains
        if self.stdp_max_iter is not None and self.stdp_cnt > self.stdp_max_iter:
            self.plasticity = False


    def __call__(self, spk_in, train=False):
        # padding 
        spk_in = np.pad(spk_in, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        # Keep records of spike input for STDP
        self.recorded_spks += spk_in
        # Output recorded spikes
        spk_out = np.zeros(self.pot.shape)
        # Convert to torch tensors
        x = torch.Tensor(spk_in).unsqueeze(0) # Add batch axis for torch conv2d
        weights = torch.Tensor(self.weights) # converts at the fly... (not so good)
        # Convolve (using torch as it is fast and easier, to be changed)
        out_conv = conv2d(x, weights, stride=self.stride).numpy()[0] # Converted to numpy
        # Update potentials
        self.pot[self.active_neurons] += out_conv[self.active_neurons]
        # Check for neurons that can spike
        output_spikes = self.pot > self.firing_threshold
        if np.any(output_spikes):
            # Generate spikes
            spk_out[output_spikes] = 1
            # Lateral inhibition for neurons in neighborhood in other channels
            # Inhibit and disable neurons with lower potential that fire
            spk_out = self.lateral_inhibition(spk_out)
            # STDP plasticity
            if train and self.plasticity:
                # Find winners (based on potential)
                winners = self.get_winners()
                # Apply STDP for each neuron winner
                for winner in winners:
                    self.stdp(winner)
            # Reset potentials and disable neurons that fire
            self.pot[spk_out == 1] = self.v_reset
            self.active_neurons[spk_out == 1] = False
        return spk_out




class SNN:
    """ 
    Spiking convolutional neural network model.
    """
    def __init__(self, input_shape):
    
        conv1 = SpikingConv(input_shape,
            out_channels=30, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=10, stdp_max_iter=None,
            adaptive_lr=True, inhibition_radius=2, v_reset=0,
        )
        
        pool1 = SpikingPool(conv1.output_shape, kernel_size=2, stride=2, padding=0)

        conv2 = SpikingConv(pool1.output_shape,
            out_channels=100, kernel_size=5, stride=1, padding=2,
            nb_winners=1, firing_threshold=1, stdp_max_iter=None,
            adaptive_lr=True, inhibition_radius=1, v_reset=0,
        )

        pool2 = SpikingPool(conv2.output_shape, kernel_size=2, stride=2, padding=0)

        self.conv_layers = [conv1, conv2]
        self.pool_layers = [pool1, pool2]
        self.output_shape = pool2.output_shape
        self.nb_trainable_layers = len(self.conv_layers)
        self.recorded_sum_spks = []


    def reset(self):
        for layer in self.conv_layers:
            layer.reset()
        for layer in self.pool_layers:
            layer.reset()


    def __call__(self, x, train_layer=None):
        self.reset()
        nb_timesteps = x.shape[0]
        output_spikes = np.zeros((nb_timesteps,) + self.output_shape)
        sum_spks = 0
        for t in range(nb_timesteps):
            spk_in = x[t].astype(np.float64)
            sum_spks += spk_in.sum()
            spk = self.conv_layers[0](spk_in, train=(train_layer==0))
            sum_spks += spk.sum()
            spk_in = self.pool_layers[0](spk)
            sum_spks += spk_in.sum()
            spk = self.conv_layers[1](spk_in, train=(train_layer==1))
            sum_spks += spk.sum()
            spk_out = self.pool_layers[1](spk)
            sum_spks += spk_out.sum()
            output_spikes[t] = spk_out
        if train_layer is None:
            self.recorded_sum_spks.append(sum_spks)
        if output_spikes.sum() == 0: print("[WARNING] No output spike recorded.")
        return output_spikes






def main(
    seed=1,
    data_prop=1, # Proportion of data to load
    nb_timesteps=15, # Number of spike bins
    epochs=[2,2], # Number of epochs per layer
    convergence_rate=0.01, # Stop training when learning convergence reaches this rate
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load encoded dataset
    X_train, y_train, X_test, y_test = load_encoded_MNIST(data_prop=data_prop, nb_timesteps=nb_timesteps)

    # Init SNN
    input_shape = X_train[0][0].shape
    snn = SNN(input_shape)

    print(f"Input shape : {X_train[0].shape} ({np.prod(X_train[0].shape)} values)")
    print(f"Output shape : {snn.output_shape} ({np.prod(snn.output_shape)} values)")
    print(f"Mean spikes count per input : {X_train.mean(0).sum()}")
    

    ### TRAINING ###
    print("\n### TRAINING ###")

    for layer in range(snn.nb_trainable_layers):
        print(f"Layer {layer+1}...")
        for epoch in range(epochs[layer]):
            print(f"\t epoch {epoch+1}")
            for x,y in zip(tqdm(X_train), y_train):
                snn(x, train_layer=layer)
                if snn.conv_layers[layer].get_learning_convergence() < convergence_rate:
                    break

    
    ### TESTING ###
    print("\n### TESTING ###")
    
    output_train_max = np.zeros((len(X_train), np.prod(snn.output_shape)))
    output_train_sum = np.zeros((len(X_train), np.prod(snn.output_shape)))
    for i,x in enumerate(tqdm(X_train)):
        spk = snn(x)
        output_train_max[i] = spk.max(0).flatten()
        output_train_sum[i] = spk.sum(0).flatten()
    
    output_test_max = np.zeros((len(X_test), np.prod(snn.output_shape)))
    output_test_sum = np.zeros((len(X_test), np.prod(snn.output_shape)))
    for i,x in enumerate(tqdm(X_test)):
        spk = snn(x)
        output_test_max[i] = spk.max(0).flatten()
        output_test_sum[i] = spk.sum(0).flatten()

    print(f"Mean total number of spikes per sample : {np.mean(snn.recorded_sum_spks)}")
    

    ### READOUT ###
    
    clf = LinearSVC(max_iter=3000, random_state=seed)
    clf.fit(output_train_max,y_train)
    y_pred = clf.predict(output_test_max)
    acc = accuracy_score(y_test,y_pred)
    print(f"Accuracy with method 1 (max) : {acc}")

    clf = LinearSVC(max_iter=3000, random_state=seed)
    clf.fit(output_train_sum,y_train)
    y_pred = clf.predict(output_test_sum)
    acc = accuracy_score(y_test,y_pred)
    print(f"Accuracy with method 2 (sum) : {acc}")




if __name__ == "__main__":
    main()
