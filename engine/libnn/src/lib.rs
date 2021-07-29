#![feature(array_methods)]

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use fast_math::sigmoid_approx;

mod fast_math;
#[cfg(test)]
mod tests;

pub type Weight = f32;

pub trait ActivationFunction {
    fn get_output(&self, x: Weight) -> Weight;

    fn derivative(&self, x: Weight) -> Weight;
}

pub struct Sigmoid;
pub static SIGMOID: Sigmoid = Sigmoid;

impl ActivationFunction for Sigmoid {
    fn get_output(&self, x: Weight) -> Weight {
        // 1. / (1. + std::f32::consts::E.powf(-x))
        sigmoid_approx(x)
    }

    fn derivative(&self, x: Weight) -> Weight {
        let y = self.get_output(x);
        y * (1. - y)
    }
}

pub struct Tanh;
pub static TANH: Tanh = Tanh;

impl ActivationFunction for Tanh {
    fn get_output(&self, x: Weight) -> Weight { x.tanh() }

    fn derivative(&self, x: Weight) -> Weight { 1. - x.tanh().powi(2) }
}

pub struct Identity;
pub static IDENTITY: Identity = Identity;

impl ActivationFunction for Identity {
    fn get_output(&self, x: Weight) -> Weight { x }

    fn derivative(&self, _x: Weight) -> Weight { 1. }
}

pub struct ReLU;
pub static RELU: ReLU = ReLU;

impl ActivationFunction for ReLU {
    fn get_output(&self, x: Weight) -> Weight {
        if x > 0. {
            x
        } else {
            0.
        }
    }

    fn derivative(&self, x: Weight) -> Weight {
        if x > 0. {
            1.
        } else {
            0.
        }
    }
}

pub trait CostFunction {
    fn get_cost(&self, error: Weight) -> Weight;

    fn derivative(&self, error: Weight) -> Weight;
}

pub struct MeanSquaredError;
pub static MEAN_SQUARED_ERROR: MeanSquaredError = MeanSquaredError;

impl CostFunction for MeanSquaredError {
    fn get_cost(&self, error: Weight) -> Weight { error * error }

    fn derivative(&self, error: Weight) -> Weight { error * 2. }
}

pub struct MeanSquaredErrorMultiplied(pub f32);

impl CostFunction for MeanSquaredErrorMultiplied {
    fn get_cost(&self, error: Weight) -> Weight { error * error * self.0 }

    fn derivative(&self, error: Weight) -> Weight { error * self.0 }
}

pub struct DenseLayer {
    pub weights: Vec<Vec<Weight>>,
    pub biases: Vec<Weight>,
    pub neuron_gradients: Vec<Weight>,
    pub activation_fn: &'static dyn ActivationFunction,
    pub outputs_before_activation: Vec<Weight>,
    pub outputs: Vec<Weight>,
}

impl DenseLayer {
    pub fn new(
        neuron_count: usize,
        input_count: usize,
        init_weights: &mut impl FnMut(usize, usize) -> Weight,
        init_biases: &mut impl FnMut(usize) -> Weight,
        activation_fn: &'static dyn ActivationFunction,
    ) -> Self {
        let mut weights = vec![vec![0.; input_count]; neuron_count];
        let mut biases = vec![0.; neuron_count];

        for neuron_ix in 0..neuron_count {
            for input_ix in 0..input_count {
                weights[neuron_ix][input_ix] = init_weights(neuron_ix, input_ix);
            }
            biases[neuron_ix] = init_biases(neuron_ix);
        }

        let neuron_gradients = vec![0.; neuron_count];

        DenseLayer {
            weights,
            biases,
            neuron_gradients,
            activation_fn,
            outputs_before_activation: vec![0.; neuron_count],
            outputs: vec![0.; neuron_count],
        }
    }

    pub fn compute_neuron_gradient(
        &self,
        neuron_output_before_activation: Weight,
        connected_output_neuron_gradient_sum: Weight,
    ) -> Weight {
        connected_output_neuron_gradient_sum * (self.activation_fn).derivative(neuron_output_before_activation)
    }

    /// Calculates the gradients for each neuron and populates `self.neuron_gradients`.
    #[inline(never)]
    pub fn compute_gradients(&mut self, output_weights: &[Vec<Weight>], gradient_of_output_neurons: &[Weight]) {
        debug_assert_eq!(output_weights.len(), gradient_of_output_neurons.len());
        // debug_assert_eq!(output_weights[0].len(), self.weights.len());

        for neuron_ix in 0..self.weights.len() {
            let output_before_activation = self.outputs_before_activation[neuron_ix];
            let mut error = 0.;

            for output_ix in 0..gradient_of_output_neurons.len() {
                let output_neuron = &output_weights[output_ix];
                let output_weight = output_neuron[neuron_ix];
                // How much the output weight we're connected to contributes to the gradient of the
                // neuron it's connected to.
                error += output_weight * gradient_of_output_neurons[output_ix];
            }

            self.neuron_gradients[neuron_ix] = self.compute_neuron_gradient(output_before_activation, error);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline(never)]
    pub fn update_weights(&mut self, inputs: &[Weight], learning_rate: Weight) {
        for (neuron_ix, &neuron_gradient) in self.neuron_gradients.iter().enumerate() {
            for (weight_ix, weight) in self.weights[neuron_ix].iter_mut().enumerate() {
                *weight += learning_rate * neuron_gradient * inputs[weight_ix];
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[inline(never)]
    pub fn update_weights(&mut self, inputs: &[Weight], learning_rate: Weight) {
        let input_count = inputs.len();
        let remainder = input_count % 4;
        let chunk_count = (input_count - remainder) / 4;
        let learning_rate_v = f32x4_splat(learning_rate);
        let inputs_ptr = inputs.as_ptr();

        for (neuron_ix, &neuron_gradient) in self.neuron_gradients.iter().enumerate() {
            let neuron_gradient_v = f32x4_splat(neuron_gradient);
            let weights_for_neuron = &mut self.weights[neuron_ix];
            let weights_for_neuron_ptr = weights_for_neuron.as_ptr();

            for chunk_ix in 0..chunk_count {
                unsafe {
                    let weights_v = v128_load(weights_for_neuron_ptr.add(chunk_ix * 4) as *const _);
                    let inputs_v = v128_load(inputs_ptr.add(chunk_ix * 4) as *const _);
                    let updated_weights = f32x4_add(
                        weights_v,
                        f32x4_mul(learning_rate_v, f32x4_mul(neuron_gradient_v, inputs_v)),
                    );
                    v128_store(weights_for_neuron_ptr.add(chunk_ix * 4) as *mut _, updated_weights)
                }
            }
            for weight_ix in (chunk_count * 4)..input_count {
                let weight = &mut weights_for_neuron[weight_ix];
                *weight += learning_rate * neuron_gradient * inputs[weight_ix];
            }
        }
    }

    #[inline(never)]
    pub fn update_biases(&mut self, learning_rate: Weight) {
        for neuron_ix in 0..self.biases.len() {
            // Each of these biases is added directly to what is fed into our activation function.
            // The impact that it will have on the output of this neuron is equal to
            // whatever the derivative of the activation function is.  We want to update the bias to
            // whatever value minimizes the gradient/error of this neuron.
            self.biases[neuron_ix] += self.neuron_gradients[neuron_ix] * learning_rate;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward_propagate(&mut self, inputs: &[Weight]) {
        debug_assert_eq!(self.weights[0].len(), inputs.len());
        for neuron_ix in 0..self.weights.len() {
            let mut weight_sum = 0.;
            for (weight_ix, &weight) in self.weights[neuron_ix].iter().enumerate() {
                let input = inputs[weight_ix];
                weight_sum += input * weight;
            }

            self.outputs_before_activation[neuron_ix] = weight_sum + self.biases[neuron_ix];
            self.outputs[neuron_ix] = (self.activation_fn).get_output(weight_sum + self.biases[neuron_ix]);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn forward_propagate(&mut self, inputs: &[Weight]) {
        debug_assert_eq!(self.weights[0].len(), inputs.len());

        let input_count = self.weights[0].len();
        let remainder = input_count % 4;
        let chunk_count = (input_count - remainder) / 4;
        let inputs_ptr = inputs.as_ptr();

        for neuron_ix in 0..self.weights.len() {
            let weights_ptr = self.weights[neuron_ix].as_ptr();
            let mut weight_sum_v = f32x4_splat(0.);
            let mut weight_sum_v_stored: [f32; 4] = [0.; 4];

            unsafe {
                for chunk_ix in 0..chunk_count {
                    let inputs = v128_load(inputs_ptr.add(4 * chunk_ix) as *const _);
                    let weights = v128_load(weights_ptr.add(4 * chunk_ix) as *const _);
                    weight_sum_v = f32x4_add(weight_sum_v, f32x4_mul(inputs, weights));
                }

                v128_store((&mut weight_sum_v_stored) as *mut _ as *mut _, weight_sum_v);
            }
            let mut weight_sum = 0.;
            for &partial_sum in weight_sum_v_stored.iter() {
                weight_sum += partial_sum;
            }
            for weight_ix in (chunk_count * 4)..input_count {
                let input = inputs[weight_ix];
                let weight = self.weights[neuron_ix][weight_ix];
                weight_sum += input * weight;
            }

            self.outputs_before_activation[neuron_ix] = weight_sum + self.biases[neuron_ix];
            self.outputs[neuron_ix] = (self.activation_fn).get_output(weight_sum + self.biases[neuron_ix]);
        }
    }
}

pub struct OutputLayer {
    pub weights: Vec<Vec<Weight>>,
    pub activation_fn: &'static dyn ActivationFunction,
    pub outputs_before_activation: Vec<Weight>,
    pub outputs: Vec<Weight>,
    pub errors: Vec<Weight>,
    pub costs: Vec<Weight>,
    pub cost_fn: &'static dyn CostFunction,
    pub neuron_gradients: Vec<Weight>,
}

impl OutputLayer {
    pub fn new(
        activation_fn: &'static dyn ActivationFunction,
        cost_fn: &'static dyn CostFunction,
        init_weights: &mut impl FnMut(usize, usize) -> Weight,
        input_count: usize,
        neuron_count: usize,
    ) -> Self {
        let mut weights: Vec<Vec<Weight>> = vec![vec![0.; input_count]; neuron_count];
        for i in 0..weights.len() {
            let neuron_weights = &mut weights[i];
            for j in 0..neuron_weights.len() {
                neuron_weights[j] = init_weights(i, j);
            }
        }

        OutputLayer {
            weights,
            activation_fn,
            outputs_before_activation: vec![0.; neuron_count],
            outputs: vec![0.; neuron_count],
            errors: vec![0.; neuron_count],
            costs: vec![0.; neuron_count],
            cost_fn,
            neuron_gradients: vec![0.; neuron_count],
        }
    }

    /// Fills `self.outputs` with output values given the outputs from the previous layer in
    /// `inputs`.
    pub fn compute(&mut self, inputs: &[Weight]) {
        for neuron_ix in 0..self.outputs.len() {
            let mut sum: Weight = 0.;
            let weights = &self.weights[neuron_ix];
            debug_assert_eq!(inputs.len(), weights.len());
            for (&weight, &input) in weights.iter().zip(inputs.iter()) {
                sum += weight * input;
            }

            // No bias on the output layer.
            self.outputs_before_activation[neuron_ix] = sum;
            self.outputs[neuron_ix] = (self.activation_fn).get_output(sum);
        }
    }

    /// Once `compute()` has been called, calculates the cost using the error for each output value
    /// and populates `self.costs.
    #[inline(never)]
    pub fn compute_costs(&mut self, expected: &[Weight]) {
        debug_assert_eq!(expected.len(), self.outputs.len());
        // Assumes that outputs have already been computed.
        for (i, &output) in self.outputs.iter().enumerate() {
            let error = expected[i] - output;
            self.errors[i] = error;
            self.costs[i] = self.cost_fn.get_cost(error);
        }
    }

    pub fn compute_neuron_gradient(&self, neuron_output_before_activation: Weight, neuron_error: Weight) -> Weight {
        (self.cost_fn).derivative(neuron_error) * (self.activation_fn).derivative(neuron_output_before_activation)
    }

    /// Once `compute_costs()` has been called, calculates the gradients for each neuron and
    /// populates `self.neuron_gradients.
    #[inline(never)]
    pub fn compute_gradients(&mut self) {
        // Assumes that costs have already been computed.
        for (neuron_ix, &neuron_error) in self.errors.iter().enumerate() {
            let output_before_activation = self.outputs_before_activation[neuron_ix];
            self.neuron_gradients[neuron_ix] = self.compute_neuron_gradient(output_before_activation, neuron_error);
        }
    }

    #[inline(never)]
    pub fn update_weights(&mut self, inputs: &[Weight], learning_rate: Weight) {
        for (neuron_ix, &neuron_gradient) in self.neuron_gradients.iter().enumerate() {
            for (weight_ix, weight) in self.weights[neuron_ix].iter_mut().enumerate() {
                *weight += learning_rate * neuron_gradient * inputs[weight_ix];
            }
        }
    }

    #[inline(never)]
    pub fn forward_propagate(&mut self, inputs: &[Weight]) {
        debug_assert_eq!(self.weights[0].len(), inputs.len());
        for neuron_ix in 0..self.weights.len() {
            let mut weight_sum = 0.;
            for (weight_ix, &weight) in self.weights[neuron_ix].iter().enumerate() {
                let input = inputs[weight_ix];
                weight_sum += input * weight;
            }
            self.outputs_before_activation[neuron_ix] = weight_sum;
            self.outputs[neuron_ix] = (self.activation_fn).get_output(weight_sum);
        }
    }
}

pub struct Network {
    pub hidden_layers: Vec<DenseLayer>,
    pub outputs: Box<OutputLayer>,
    pub learning_rate: Weight,
}

impl Network {
    pub fn forward_propagate(&mut self, inputs: &[Weight]) {
        let mut inputs: &[Weight] = inputs;
        for layer in &mut self.hidden_layers {
            layer.forward_propagate(inputs);
            inputs = &layer.outputs;
        }

        self.outputs.forward_propagate(inputs);
    }

    pub fn train_one_example(&mut self, example: &[Weight], expected: &[f32], learning_rate: f32) {
        // Run the example all the way through the network, populating outputs in the output layer.
        self.forward_propagate(example);

        // Compute gradients + costs for the output layer based off the generated outputs
        self.outputs.compute_costs(expected);
        self.outputs.compute_gradients();

        // Then compute gradients for the hidden layers
        let mut output_weights = self.outputs.weights.as_slice();
        let mut gradient_of_output_neurons = self.outputs.neuron_gradients.as_slice();
        for hidden_layer in self.hidden_layers.iter_mut().rev() {
            hidden_layer.compute_gradients(output_weights, gradient_of_output_neurons);
            output_weights = hidden_layer.weights.as_slice();
            gradient_of_output_neurons = &hidden_layer.neuron_gradients.as_slice();
        }

        // Using the gradients computed before, update weights on the output layer
        let inputs = self.hidden_layers.last().unwrap().outputs.as_slice();
        self.outputs.update_weights(inputs, self.learning_rate);

        // then update weights + biases for all hidden layers
        for hidden_layer_ix in (0..self.hidden_layers.len()).rev() {
            let inputs = if hidden_layer_ix == 0 {
                example
            } else {
                // I don't care about lifetimes here, we only mutate the output
                let slice = self.hidden_layers[hidden_layer_ix - 1].outputs.as_slice();
                unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) }
            };
            let hidden_layer = &mut self.hidden_layers[hidden_layer_ix];
            hidden_layer.update_weights(inputs, self.learning_rate);
            hidden_layer.update_biases(learning_rate);
        }

        // That's it, we've successfully "learned"
    }

    pub fn compute<'a>(&'a mut self, inputs: &[Weight]) -> &'a [Weight] {
        self.forward_propagate(inputs);
        &self.outputs.outputs
    }
}
