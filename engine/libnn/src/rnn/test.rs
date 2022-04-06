use rand::Rng;

use super::{RecurrentLayer, RecurrentNetwork};
use crate::{OutputLayer, Weight, IDENTITY, MEAN_SQUARED_ERROR};

fn build_test_network(input_size: usize, output_size: usize, state_size: usize) -> RecurrentNetwork {
    let mut init_weights = |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(0.0, 0.1) };
    let mut init_biases = |_output_ix: usize| -> Weight { 0. };

    RecurrentNetwork {
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
    }
}

/// This is as simple as it gets.  Optimize the weights of the output tree towards zero for all inputs.
#[test]
fn rnn_sanity_output_zero() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 1;
    let learning_rate = 0.25;
    let mut network = build_test_network(input_size, output_size, state_size);

    let training_sequence = vec![vec![1.], vec![0.5]];
    let expected_outputs = vec![vec![0.0], vec![0.0]];

    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..10 {
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        println!("[{}] cost: {}", i, new_cost);
        println!("[{}] outputs: {:?}", i, network.outputs);
        last_iter_cost = new_cost;
    }
    assert!(last_iter_cost < 0.0001);
}

/// Output current value in the sequence
#[test]
fn rnn_sanity_output_identity() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 1;
    let learning_rate = 0.05;
    let mut network = build_test_network(input_size, output_size, state_size);

    let training_sequence = vec![vec![1.], vec![0.5], vec![1.], vec![0.5]];
    let expected_outputs = vec![vec![1.], vec![0.5], vec![1.], vec![0.5]];

    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..300 {
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        println!("[{}] cost: {}", i, new_cost);
        println!("[{}] outputs: {:?}", i, network.outputs);
        last_iter_cost = new_cost;
    }
    assert!(last_iter_cost < 0.0001);
}

/// Output previous value in the sequence
#[test]
fn rnn_sanity_output_last_value() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 1;
    let learning_rate = 0.02;
    let mut network = build_test_network(input_size, output_size, state_size);

    let training_sequence = vec![vec![1.], vec![0.5], vec![1.], vec![0.5], vec![0.3], vec![0.2], vec![0.]];
    let expected_outputs = vec![vec![0.], vec![1.], vec![0.5], vec![1.], vec![0.5], vec![0.3], vec![0.2]];

    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..2000 {
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        println!("[{}] cost: {}", i, new_cost);
        println!("[{}] outputs: {:?}", i, network.outputs);
        last_iter_cost = new_cost;
    }
    assert!(last_iter_cost < 0.001);

    println!(
        "RECURRENT WEIGHTS: {:?}",
        network.recurrent_layer.recurrent_tree.weights
    );
    println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
    println!("FINAL STATE: {:?}", network.recurrent_layer.state);
}

/// Output value seen 2 steps ago
#[test]
fn rnn_sanity_output_2_steps_back() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 2;
    let learning_rate = 0.01;
    let mut network = build_test_network(input_size, output_size, state_size);

    let training_sequence = vec![
        vec![1.],
        vec![0.5],
        vec![1.],
        vec![0.5],
        vec![0.3],
        vec![0.2],
        vec![0.],
        vec![0.9],
    ];
    let expected_outputs = vec![
        vec![0.],
        vec![0.],
        vec![1.],
        vec![0.5],
        vec![1.],
        vec![0.5],
        vec![0.3],
        vec![0.2],
    ];

    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..2000 {
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        println!("[{}] cost: {}", i, new_cost);
        println!("[{}] outputs: {:?}", i, network.outputs);
        last_iter_cost = new_cost;
    }
    assert!(last_iter_cost < 0.001);

    println!(
        "RECURRENT WEIGHTS: {:?}",
        network.recurrent_layer.recurrent_tree.weights
    );
    println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
    println!("FINAL STATE: {:?}", network.recurrent_layer.state);

    // Get the next two predictions
    let prediction_1 = network.predict(&[vec![0.9]]).to_owned();
    let prediction_2 = network.predict(&[vec![0.1]]).to_owned();
    println!("PREDICTION 1: {:?}", prediction_1);
    println!("PREDICTION 2: {:?}", prediction_2);
}

#[test]
fn rnn_memory_conditional() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 4;
    let learning_rate = 0.01;
    let mut network = build_test_network(input_size, output_size, state_size);

    // fn gen_training_data() -> Vec<()
}
