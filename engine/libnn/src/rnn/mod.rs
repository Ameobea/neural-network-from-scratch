use crate::{ActivationFunction, DenseLayer, OutputLayer, Weight};

#[cfg(test)]
mod test;

pub struct RecurrentLayer {
    pub state: Vec<Weight>,
    pub inner: DenseLayer,
    pub combined_inputs_scratch: Vec<Weight>,
    pub sequence_inputs: Vec<Vec<Weight>>,
    pub prev_states: Vec<Vec<Weight>>,
    /// Computed gradients for each step of the sequence
    pub computed_gradients: Vec<Vec<Weight>>,
}

impl RecurrentLayer {
    pub fn new(
        neuron_count: usize,
        input_count: usize,
        init_weights: &mut impl FnMut(usize, usize) -> Weight,
        init_biases: &mut impl FnMut(usize) -> Weight,
        activation_fn: &'static dyn ActivationFunction,
        state_size: usize,
    ) -> Self {
        // State always initialized to all zeros for now
        let state = vec![0.; state_size];

        RecurrentLayer {
            state,
            inner: DenseLayer::new(
                neuron_count + state_size,
                input_count + state_size,
                init_weights,
                init_biases,
                activation_fn,
            ),
            combined_inputs_scratch: vec![0.; input_count + state_size],
            sequence_inputs: Vec::new(),
            prev_states: Vec::new(),
            computed_gradients: Vec::new(),
        }
    }

    pub fn reset(&mut self) { self.state.fill(0.); }

    pub fn forward_propagate(&mut self, inputs: &[Weight], index_in_sequence: usize) {
        // Build combined inputs from state, inputs
        self.combined_inputs_scratch[..self.state.len()].copy_from_slice(&self.state);
        self.combined_inputs_scratch[self.state.len()..].copy_from_slice(inputs);
        self.inner.forward_propagate(&self.combined_inputs_scratch);

        // Save inputs + previous state for backpropagation
        if let Some(slot) = self.prev_states.get_mut(index_in_sequence) {
            slot.copy_from_slice(&self.state);
        } else {
            self.prev_states.push(self.state.clone());
        }
        if let Some(slot) = self.sequence_inputs.get_mut(index_in_sequence) {
            slot.copy_from_slice(inputs);
        } else {
            self.sequence_inputs.push(inputs.to_vec());
        }

        // Update state
        let state_size = self.state.len();
        self.state.copy_from_slice(&self.inner.outputs[..state_size]);
    }

    /// Gets the part of the output that is not fed back into the state - the part which is passed on to the next layer.
    pub fn get_outputs(&self) -> &[Weight] { &self.inner.outputs[self.state.len()..] }

    pub fn compute_gradients(&mut self, output_weights: &[Vec<Weight>], gradient_of_output_neurons: &[Vec<Weight>]) {
        // Iterate backwards through the sequence, computing gradients for each step.
        //
        // For the last step, the gradient can be computed using the provided weights and gradient of the neurons in the
        // next layer we're connected to.
        //
        // The gradient of the "recurrent" neurons that output the state is set to zero because we don't care about the
        // value of the state at the end of the sequence.

        // TODO: Probably want to move these to scratch to avoid having to re-allocate
        let state_size = self.state.len();
        let mut combined_output_weights = Vec::with_capacity(state_size + output_weights.len());
        let mut combined_gradient_of_output_neurons = Vec::with_capacity(state_size + gradient_of_output_neurons.len());
        // Gradient of recurrent neurons is zero to start
        for _ in 0..state_size {
            combined_output_weights.push(vec![0.; output_weights[0].len()]);
            combined_gradient_of_output_neurons.push(0.);
        }
        combined_output_weights.extend_from_slice(output_weights);
        combined_gradient_of_output_neurons.extend_from_slice(gradient_of_output_neurons.last().unwrap());
        debug_assert_eq!(combined_output_weights.len(), combined_gradient_of_output_neurons.len());
        debug_assert_eq!(combined_output_weights.len(), self.inner.weights.len());

        // These are in reverse order because we're iterating backwards through the sequence
        let mut computed_gradients = Vec::with_capacity(self.prev_states.len());

        self.inner
            .compute_gradients(&combined_output_weights, &combined_gradient_of_output_neurons);
        let mut last_step_gradients = &self.inner.neuron_gradients;
        debug_assert_eq!(combined_gradient_of_output_neurons.len(), last_step_gradients.len());
        computed_gradients.push(last_step_gradients.clone());
        last_step_gradients = computed_gradients.last().unwrap();

        // Update the combined output weights with our own weights since they are now impactful
        for (i, weights) in self.inner.weights.iter().enumerate() {
            debug_assert_eq!(weights.len(), combined_output_weights[i].len());
            combined_output_weights[i].copy_from_slice(&weights);
        }

        // Continue iterating backwards through the sequence, computing gradients for each step using the gradients of
        // the next step.
        for i in (0..self.prev_states.len()).rev().skip(1) {
            combined_gradient_of_output_neurons[..state_size].copy_from_slice(last_step_gradients);
            combined_gradient_of_output_neurons[state_size..].copy_from_slice(&gradient_of_output_neurons[i]);
            self.inner
                .compute_gradients(&combined_output_weights, &last_step_gradients);
            last_step_gradients = &self.inner.neuron_gradients;
            debug_assert_eq!(combined_gradient_of_output_neurons.len(), last_step_gradients.len());
            computed_gradients.push(last_step_gradients.clone());
            last_step_gradients = computed_gradients.last().unwrap();
        }

        computed_gradients.reverse();
        self.computed_gradients = computed_gradients;
    }

    pub fn update_weights(&mut self, learning_rate: Weight) {
        let step_count = self.prev_states.len();
        self.combined_inputs_scratch.fill(0.);
        for step_ix in 0..step_count {
            // TODO: Don't need to copy inputs into a buffer; can just use the slices directly
            if step_ix == 0 {
                // Internal state is initialized to 0 at the first step of the sequence
            } else {
                // Internal state is initialized to the state from the previous step
                self.combined_inputs_scratch[..self.state.len()].copy_from_slice(&self.prev_states[step_ix]);
            }
            self.combined_inputs_scratch[self.state.len()..].copy_from_slice(&self.sequence_inputs[step_ix]);

            // Maybe we should accumulate the gradients into a scratch buffer instead of adding multiple times?
            for (neuron_ix, &neuron_gradient) in self.computed_gradients[step_ix].iter().enumerate() {
                for (weight_ix, weight) in self.inner.weights[neuron_ix].iter_mut().enumerate() {
                    *weight += learning_rate * neuron_gradient * self.combined_inputs_scratch[weight_ix];
                }
            }
        }
    }

    pub fn update_biases(&mut self, learning_rate: Weight) {
        // TODO
    }
}

pub struct RecurrentNetwork {
    pub recurrent_layer: RecurrentLayer,
    pub output_layer: Box<OutputLayer>,
    pub outputs: Vec<Vec<Weight>>,
}

impl RecurrentNetwork {
    /// Returns (total_cost, output_gradients)
    pub fn forward_propagate(
        &mut self,
        sequence: &[Vec<Weight>],
        expected_sequence: Option<&[Vec<Weight>]>,
    ) -> (f32, Vec<Vec<f32>>) {
        let mut output_gradients = Vec::new();
        let mut total_costs = 0.;

        for (step_ix, example) in sequence.iter().enumerate() {
            self.recurrent_layer.forward_propagate(example, step_ix);
            self.output_layer.forward_propagate(self.recurrent_layer.get_outputs());

            match self.outputs.get_mut(step_ix) {
                Some(slot) => slot.copy_from_slice(&self.output_layer.outputs),
                None => self.outputs.push(self.output_layer.outputs.clone()),
            }

            if let Some(expected_sequence) = expected_sequence {
                self.output_layer.compute_costs(&expected_sequence[step_ix]);
                total_costs += self.output_layer.costs.iter().fold(0., |acc, cost| acc + *cost);
                self.output_layer.compute_gradients();
                output_gradients.push(self.output_layer.neuron_gradients.clone());
            }
        }

        (total_costs, output_gradients)
    }

    /// Returns the average cost of the output before updating weights.  It would be better to compute again after, but
    /// that would be too expensive
    pub fn train_one_sequence(
        &mut self,
        sequence: &[Vec<Weight>],
        expected_sequence: &[Vec<Weight>],
        learning_rate: Weight,
    ) -> Weight {
        // Run the sequence all the way through the network, populating outputs and keeping track of output layer
        // gradients for each step
        let (total_cost, output_gradients) = self.forward_propagate(sequence, Some(expected_sequence));

        // The compute gradients of the recurrent layer for each step of the sequence
        self.recurrent_layer
            .compute_gradients(&self.output_layer.weights, &output_gradients);

        // TODO: Update the weights of the output layer

        // Update weights + biases of the recurrent layer
        self.recurrent_layer.update_weights(learning_rate);
        self.recurrent_layer.update_biases(learning_rate);

        // That's it, we've successfully "learned"
        total_cost / self.output_layer.costs.len() as Weight
    }
}
