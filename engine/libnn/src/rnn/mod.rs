use crate::{ActivationFunction, DenseLayer, OutputLayer, Weight};

#[cfg(test)]
mod test;

pub struct RecurrentTree {
    layers: Vec<DenseLayer>,
    neuron_gradients_by_layer_by_step_ix: Vec<Vec<Vec<Weight>>>,
    layer_outputs_by_layer_by_step_ix: Vec<Vec<Vec<Weight>>>,
}

impl RecurrentTree {
    pub fn new(mut def: Vec<RecurrentTreeLayerDef>) -> Self {
        Self {
            layers: def.iter_mut().map(|def| def.build_layer()).collect(),
            neuron_gradients_by_layer_by_step_ix: Vec::new(),
            layer_outputs_by_layer_by_step_ix: Vec::new(),
        }
    }

    pub fn forward_propagate(&mut self, inputs: &[f32], step_ix: usize) {
        let mut inputs = inputs;
        for layer in &mut self.layers {
            layer.forward_propagate(inputs);
            inputs = &layer.outputs;
        }

        while self.layer_outputs_by_layer_by_step_ix.len() <= step_ix {
            self.layer_outputs_by_layer_by_step_ix.push(Vec::new());
        }
        self.layer_outputs_by_layer_by_step_ix[step_ix] =
            self.layers.iter().map(|layer| layer.outputs.clone()).collect();
    }

    pub fn first_layer_weights(&self) -> &[Vec<f32>] { &self.layers.first().unwrap().weights }

    pub fn all_layer_weights(&self) -> Vec<Vec<Vec<Weight>>> {
        self.layers.iter().map(|layer| layer.weights.clone()).collect()
    }

    pub fn all_layer_biases(&self) -> Vec<Vec<f32>> { self.layers.iter().map(|layer| layer.biases.clone()).collect() }

    pub fn outputs(&self) -> &[f32] { &self.layers.last().unwrap().outputs }

    pub fn compute_gradients(
        &mut self,
        output_weights: &[Vec<f32>],
        gradient_of_output_neurons: &[f32],
        step_ix: usize,
        accumulate: bool,
    ) {
        let mut output_weights = output_weights;
        let mut gradient_of_output_neurons = gradient_of_output_neurons;
        for layer in self.layers.iter_mut().rev() {
            layer.compute_gradients(output_weights, gradient_of_output_neurons);
            output_weights = layer.weights.as_slice();
            gradient_of_output_neurons = &layer.neuron_gradients.as_slice();
        }

        while self.neuron_gradients_by_layer_by_step_ix.len() <= step_ix {
            self.neuron_gradients_by_layer_by_step_ix.push(Vec::new());
        }

        if accumulate {
            for (layer_ix, gradients_for_layer) in self.neuron_gradients_by_layer_by_step_ix[step_ix]
                .iter_mut()
                .enumerate()
            {
                for (neuron_ix, gradient) in gradients_for_layer.iter_mut().enumerate() {
                    *gradient += self.layers[layer_ix].neuron_gradients[neuron_ix];
                    *gradient /= 2.;
                }
            }
        } else {
            self.neuron_gradients_by_layer_by_step_ix[step_ix] =
                self.layers.iter().map(|layer| layer.neuron_gradients.clone()).collect();
        }
    }

    pub fn neuron_gradients(&self) -> &[f32] { &self.layers.first().unwrap().neuron_gradients }

    pub fn update_weights_and_biases(
        &mut self,
        inputs: &[f32],
        learning_rate: f32,
        step_ix: usize,
        sequence_len: usize,
    ) {
        if step_ix == sequence_len - 1 {
            // Recurrent tree has no gradient at the last step in the sequence since its output is not used
            return;
        }

        for layer_ix in (0..self.layers.len()).rev() {
            let inputs = if layer_ix == 0 {
                inputs
            } else {
                // I don't care about lifetimes here, we only mutate the output
                let slice = self.layer_outputs_by_layer_by_step_ix[step_ix][layer_ix - 1].as_slice();
                unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) }
            };
            let layer = &mut self.layers[layer_ix];

            layer
                .neuron_gradients
                .copy_from_slice(self.neuron_gradients_by_layer_by_step_ix[step_ix][layer_ix].as_slice());

            layer.update_weights(inputs, learning_rate);
            // layer.update_biases(learning_rate);
        }
    }
}

pub struct RecurrentLayer {
    pub state: Vec<Weight>,
    pub recurrent_tree: RecurrentTree,
    pub output_tree: DenseLayer,
    pub combined_inputs_scratch: Vec<Weight>,
    pub sequence_inputs: Vec<Vec<Weight>>,
    pub prev_states: Vec<Vec<Weight>>,
    // Computed gradients for each step of the sequence
    pub computed_recurrent_gradients: Vec<Vec<Weight>>,
    pub computed_output_gradients: Vec<Vec<Weight>>,
}

pub struct RecurrentTreeLayerDef {
    /// Must be the same as the output count of the previous layer.  If this is the first layer of the recurrent tree,
    /// must be equal to input size of the whole stack + output size of the last layer
    pub input_count: usize,
    /// Must be the same as the input count of the next layer
    pub output_count: usize,
    pub init_weights: Box<dyn FnMut(usize, usize) -> Weight>,
    pub init_biases: Box<dyn FnMut(usize) -> Weight>,
    pub activation_fn: &'static dyn ActivationFunction,
}

impl RecurrentTreeLayerDef {
    pub fn build_layer(&mut self) -> DenseLayer {
        DenseLayer::new(
            self.output_count,
            self.input_count,
            &mut self.init_weights,
            &mut self.init_biases,
            self.activation_fn,
        )
    }
}

impl RecurrentLayer {
    pub fn new(
        output_count: usize,
        input_count: usize,
        recurrent_tree_def: Vec<RecurrentTreeLayerDef>,
        init_output_weights: &mut impl FnMut(usize, usize) -> Weight,
        init_output_biases: &mut impl FnMut(usize) -> Weight,
        output_activation_fn: &'static dyn ActivationFunction,
        state_size: usize,
    ) -> Self {
        // State always initialized to all zeros for now
        let state = vec![0.; state_size];

        assert!(recurrent_tree_def.len() > 0);
        assert_eq!(recurrent_tree_def[0].input_count, input_count + state_size);
        assert_eq!(recurrent_tree_def.last().unwrap().output_count, state_size);

        RecurrentLayer {
            state,
            recurrent_tree: RecurrentTree::new(recurrent_tree_def),
            output_tree: DenseLayer::new(
                output_count,
                input_count + state_size,
                init_output_weights,
                init_output_biases,
                output_activation_fn,
            ),
            combined_inputs_scratch: vec![0.; input_count + state_size],
            sequence_inputs: Vec::new(),
            prev_states: Vec::new(),
            computed_recurrent_gradients: Vec::new(),
            computed_output_gradients: Vec::new(),
        }
    }

    pub fn reset(&mut self) { self.state.fill(0.); }

    pub fn forward_propagate(&mut self, inputs: &[Weight], index_in_sequence: usize) {
        // Build combined inputs from state, inputs
        self.combined_inputs_scratch[..self.state.len()].copy_from_slice(&self.state);
        self.combined_inputs_scratch[self.state.len()..].copy_from_slice(inputs);

        self.output_tree.forward_propagate(&self.combined_inputs_scratch);
        self.recurrent_tree
            .forward_propagate(&self.combined_inputs_scratch, index_in_sequence);

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
        // TODO: Can eventually avoid copying this buffer
        self.state.copy_from_slice(&self.recurrent_tree.outputs());
    }

    /// Gets the part of the output that is not fed back into the state - the part which is passed on to the next layer.
    pub fn get_outputs(&self) -> &[Weight] { &self.output_tree.outputs }

    pub fn compute_gradients(
        &mut self,
        output_output_weights: &[Vec<Weight>],
        output_gradient_of_output_neurons: &[Vec<Weight>],
        sequence_len: usize,
    ) {
        // Iterate backwards through the sequence, computing gradients for each step.
        //
        // For the last step, the gradient can be computed using the provided weights and gradient of the neurons in the
        // next layer we're connected to.
        //
        // The gradient of the "recurrent" neurons that output the state is set to zero because we don't care about the
        // value of the state at the end of the sequence.

        // These are in reverse order because we're iterating backwards through the sequence
        self.computed_output_gradients = Vec::with_capacity(self.prev_states.len());
        self.computed_recurrent_gradients = Vec::with_capacity(self.prev_states.len());

        let recurrent_tree_to_recurrent_tree_connected_weights: Vec<_> = self
            .recurrent_tree
            .first_layer_weights()
            .iter()
            .map(|weights| weights[..self.state.len()].to_owned())
            .collect();
        let recurrent_tree_to_output_tree_connected_weights: Vec<_> = self
            .output_tree
            .weights
            .iter()
            .map(|weights| weights[..self.state.len()].to_owned())
            .collect();

        // Output -> Output gradients are computed using the provided gradient of the next external layer
        self.output_tree.compute_gradients(
            output_output_weights,
            &output_gradient_of_output_neurons[sequence_len - 1],
        );
        self.computed_output_gradients
            .push(self.output_tree.neuron_gradients.clone());

        // Recurrent -> Output gradients are 0 since the recurrent neuron is basically unconnected at the last step of
        // the sequence.
        //
        // Recurrent -> Recurrent gradients are also 0 for the last step since we don't care about the state at the end
        // of the sequence.
        self.computed_recurrent_gradients
            .push(vec![0.; self.recurrent_tree.neuron_gradients().len()]);

        // Continue iterating backwards through the sequence, computing gradients for each step using the gradients of
        // the step after it.
        for step_ix in (0..sequence_len).rev().skip(1) {
            // Recurrent -> Output gradients are computed using the provided gradient of the next external layer
            self.recurrent_tree.compute_gradients(
                &recurrent_tree_to_output_tree_connected_weights,
                self.computed_output_gradients.last().unwrap(),
                step_ix,
                false,
            );
            let mut recurrent_to_output_gradients = self.recurrent_tree.neuron_gradients().to_owned();

            // Recurrent -> Recurrent gradients are computed using the gradients of the step after it and the parts of
            // its own weights that are connected to its own outputs.
            self.recurrent_tree.compute_gradients(
                &recurrent_tree_to_recurrent_tree_connected_weights,
                self.computed_recurrent_gradients.last().unwrap(),
                step_ix,
                true,
            );

            // Combine the gradients
            debug_assert_eq!(
                recurrent_to_output_gradients.len(),
                self.recurrent_tree.neuron_gradients().len()
            );
            for i in 0..recurrent_to_output_gradients.len() {
                recurrent_to_output_gradients[i] += self.recurrent_tree.neuron_gradients()[i];
                recurrent_to_output_gradients[i] /= 2.;
            }
            self.computed_recurrent_gradients.push(recurrent_to_output_gradients);

            // Output -> Output gradients are computed using the provided gradient of the next external layer
            self.output_tree
                .compute_gradients(output_output_weights, &output_gradient_of_output_neurons[step_ix]);
            self.computed_output_gradients
                .push(self.output_tree.neuron_gradients.clone());
        }

        debug_assert_eq!(
            self.computed_output_gradients.len(),
            self.computed_recurrent_gradients.len()
        );

        self.computed_output_gradients.reverse();
        self.computed_recurrent_gradients.reverse();
    }

    pub fn update_weights(&mut self, learning_rate: Weight, sequence_len: usize) {
        self.combined_inputs_scratch.fill(0.);

        for step_ix in 0..sequence_len {
            // TODO: Don't need to copy inputs into a buffer; can just use the slices directly
            if step_ix == 0 {
                // Internal state is initialized to 0 at the first step of the sequence
            } else {
                // Internal state is initialized to the state from the previous step
                self.combined_inputs_scratch[..self.state.len()].copy_from_slice(&self.prev_states[step_ix]);
            }
            self.combined_inputs_scratch[self.state.len()..].copy_from_slice(&self.sequence_inputs[step_ix]);

            // Maybe we should accumulate the gradients into a scratch buffer instead of adding multiple times?
            for (neuron_ix, &neuron_gradient) in self.computed_output_gradients[step_ix].iter().enumerate() {
                for (weight_ix, weight) in self.output_tree.weights[neuron_ix].iter_mut().enumerate() {
                    *weight += learning_rate * neuron_gradient * self.combined_inputs_scratch[weight_ix];
                }
            }

            self.recurrent_tree.update_weights_and_biases(
                &self.combined_inputs_scratch,
                learning_rate,
                step_ix,
                sequence_len,
            );
        }
    }

    pub fn update_biases(&mut self, learning_rate: Weight, sequence_len: usize) {
        for step_ix in 0..sequence_len {
            for neuron_ix in 0..self.output_tree.biases.len() {
                // Each of these biases is added directly to what is fed into our activation function.
                // The impact that it will have on the output of this neuron is equal to
                // whatever the derivative of the activation function is.  We want to update the bias to
                // whatever value minimizes the gradient/error of this neuron.
                self.output_tree.biases[neuron_ix] +=
                    self.computed_output_gradients[step_ix][neuron_ix] * learning_rate;
            }
        }
    }
}

pub struct RecurrentNetwork {
    pub recurrent_layer: RecurrentLayer,
    pub output_layer: Box<OutputLayer>,
    pub recurrent_layer_outputs: Vec<Vec<Weight>>,
    pub outputs: Vec<Vec<Weight>>,
}

impl RecurrentNetwork {
    /// Returns (total_cost, output_gradients)
    pub fn forward_propagate(
        &mut self,
        sequence: &[Vec<Weight>],
        expected_sequence: Option<&[Option<Vec<Weight>>]>,
    ) -> (f32, Vec<Vec<f32>>) {
        // Reset state in recurrent layer to its default value
        self.recurrent_layer.reset();

        let mut output_gradients = Vec::new();
        let mut total_costs = 0.;

        for (step_ix, example) in sequence.iter().enumerate() {
            self.recurrent_layer.forward_propagate(example, step_ix);
            self.output_layer.forward_propagate(self.recurrent_layer.get_outputs());

            match self.outputs.get_mut(step_ix) {
                Some(slot) => slot.copy_from_slice(&self.output_layer.outputs),
                None => self.outputs.push(self.output_layer.outputs.clone()),
            }
            match self.recurrent_layer_outputs.get_mut(step_ix) {
                Some(slot) => slot.copy_from_slice(&self.recurrent_layer.get_outputs()),
                None => self
                    .recurrent_layer_outputs
                    .push(self.recurrent_layer.get_outputs().to_owned()),
            }

            if let Some(expected_sequence) = expected_sequence {
                let gradients = if let Some(expected_output) = &expected_sequence[step_ix] {
                    self.output_layer.compute_costs(expected_output);
                    total_costs += self.output_layer.costs.iter().fold(0., |acc, cost| acc + *cost);
                    self.output_layer.compute_gradients();
                    self.output_layer.neuron_gradients.clone()
                } else {
                    vec![0.; self.output_layer.neuron_gradients.len()]
                };
                output_gradients.push(gradients);
            }
        }

        (total_costs, output_gradients)
    }

    /// Returns the average cost of the output before updating weights.  It would be better to compute again after, but
    /// that would be too expensive
    pub fn train_one_sequence(
        &mut self,
        sequence: &[Vec<Weight>],
        expected_sequence: &[Option<Vec<Weight>>],
        learning_rate: Weight,
    ) -> Weight {
        assert_eq!(sequence.len(), expected_sequence.len());

        // Run the sequence all the way through the network, populating outputs and keeping track of output layer
        // gradients for each step
        let (total_cost, output_gradients) = self.forward_propagate(sequence, Some(expected_sequence));

        // The compute gradients of the recurrent layer for each step of the sequence
        self.recurrent_layer
            .compute_gradients(&self.output_layer.weights, &output_gradients, sequence.len());

        assert_eq!(self.outputs.len(), self.recurrent_layer_outputs.len());
        // TODO: Re-enable
        for i in 0..self.outputs.len() {
            let inputs_to_output_layer = &self.recurrent_layer_outputs[i];
            self.output_layer
                .update_weights(&inputs_to_output_layer, learning_rate * (1. / sequence.len() as f32));
        }

        // Update weights + biases of the recurrent layer
        self.recurrent_layer.update_weights(learning_rate, sequence.len());
        // self.recurrent_layer.update_biases(learning_rate, sequence.len());

        // That's it, we've successfully "learned"
        (total_cost / self.output_layer.costs.len() as Weight) / sequence.len() as Weight
    }

    pub fn predict(&mut self, sequence: &[Vec<Weight>]) -> &[Vec<Weight>] {
        self.forward_propagate(sequence, None);
        &self.outputs[..sequence.len()]
    }
}
