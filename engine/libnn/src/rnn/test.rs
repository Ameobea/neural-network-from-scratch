use rand::Rng;

use super::{RecurrentLayer, RecurrentNetwork};
use crate::{OutputLayer, Weight, IDENTITY, MEAN_SQUARED_ERROR};

#[test]
fn rnn_sanity() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 2;
    let learning_rate = 0.25;

    let mut init_weights = |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(-1.0, 1.0) };
    let mut init_biases = |_output_ix: usize| -> Weight { 0. };

    let mut network = RecurrentNetwork {
        recurrent_layer: RecurrentLayer::new(
            output_size,
            input_size,
            &mut init_weights,
            &mut init_biases,
            &IDENTITY,
            state_size,
        ),
        output_layer: Box::new(OutputLayer::new(
            &IDENTITY,
            &MEAN_SQUARED_ERROR,
            &mut |_, _| 1.,
            input_size,
            output_size,
        )),
        outputs: Vec::new(),
    };

    let training_sequence = vec![vec![1.], vec![0.], vec![1.], vec![0.]];
    let expected_outputs = vec![vec![0.], vec![1.], vec![0.], vec![1.]];

    let (initial_total_cost, output_gradients) = network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let post_training_costs = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
    println!("post-training cost: {}", post_training_costs);
    println!("post-training outputs: {:?}", network.outputs);
    assert!(post_training_costs < initial_total_cost)
}
