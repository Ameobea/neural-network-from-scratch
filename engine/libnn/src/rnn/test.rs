use rand::Rng;

use super::{RecurrentLayer, RecurrentNetwork, RecurrentTreeLayerDef};
use crate::{OutputLayer, Weight, IDENTITY, MEAN_SQUARED_ERROR};

fn build_test_network(input_size: usize, output_size: usize, state_size: usize) -> RecurrentNetwork {
    let init_recurrent_weights =
        |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(0.0, 0.1) };
    let init_recurrent_biases = |_output_ix: usize| -> Weight { 0. };
    let recurrent_activation_fn = &IDENTITY;

    let mut init_output_weights =
        |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(0.0, 0.1) };
    let mut init_output_biases = |_output_ix: usize| -> Weight { 0. };
    let output_activation_fn = &IDENTITY;

    let recurrent_layer_def = vec![RecurrentTreeLayerDef {
        input_count: input_size + state_size,
        output_count: state_size,
        init_weights: Box::new(init_recurrent_weights),
        init_biases: Box::new(init_recurrent_biases),
        activation_fn: recurrent_activation_fn,
    }];

    RecurrentNetwork {
        recurrent_layer: RecurrentLayer::new(
            output_size,
            input_size,
            recurrent_layer_def,
            &mut init_output_weights,
            &mut init_output_biases,
            output_activation_fn,
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
        recurrent_layer_outputs: Vec::new(),
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
    let expected_outputs = vec![Some(vec![0.0]), Some(vec![0.0])];

    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..100 {
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

    fn gen_training_data() -> (Vec<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        let sequence_len = 4;
        let mut training_sequence = Vec::with_capacity(sequence_len);

        for _ in 0..sequence_len {
            training_sequence.push(vec![rand::thread_rng().gen_range(-1., 1.)]);
        }
        let expected_sequence = training_sequence.iter().map(|v| Some(v.clone())).collect();

        (training_sequence, expected_sequence)
    }

    let (training_sequence, expected_outputs) = gen_training_data();
    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..2000 {
        let (training_sequence, expected_outputs) = gen_training_data();
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        println!("[{}] cost: {}", i, new_cost);
        println!("[{}] outputs: {:?}", i, network.outputs);
        last_iter_cost = new_cost;
    }
    assert!(last_iter_cost < 0.001);
}

/// Output previous value in the sequence
#[test]
fn rnn_sanity_output_last_value() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 4;
    let learning_rate = 0.05;
    let mut network = build_test_network(input_size, output_size, state_size);

    fn gen_training_data() -> (Vec<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        let sequence_len = rand::thread_rng().gen_range(3usize, 4usize);
        let mut training_sequence = Vec::with_capacity(sequence_len);
        let mut expected_outputs = Vec::with_capacity(sequence_len);

        for i in 0..sequence_len {
            training_sequence.push(vec![rand::thread_rng().gen_range(-1., 1.)]);
            if i == 0 {
                expected_outputs.push(None);
            } else {
                expected_outputs.push(Some(training_sequence[i - 1].clone()));
            }
        }

        (training_sequence, expected_outputs)
    }

    let (training_sequence, expected_outputs) = gen_training_data();
    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut cost = initial_total_cost;
    for i in 0..5000 {
        let (training_sequence, expected_outputs) = gen_training_data();
        cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        if cost.is_nan() {
            panic!();
        }
        println!("");
        println!("[{}] cost: {}", i, cost);
        println!("[{}] inputs: {:?}", i, training_sequence.as_slice());
        println!("[{}] outputs: {:?}", i, &network.outputs[..training_sequence.len()]);
        println!(
            "[{}] expected: {:?}",
            i,
            expected_outputs
                .clone()
                .into_iter()
                .map(|o| o.unwrap_or_else(|| vec![-0.]))
                .collect::<Vec<_>>()
                .as_slice()
        );
        // println!(
        //     "\nRECURRENT WEIGHTS: {:?}",
        //     network.recurrent_layer.recurrent_tree.all_layer_weights()
        // );
        // println!(
        //     "RECURRENT BIASES: {:?}",
        //     network.recurrent_layer.recurrent_tree.all_layer_biases()
        // );
        // println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
        // println!("OUTPUT BIASES: {:?}", network.recurrent_layer.output_tree.biases);
        // println!("FINAL STATE: {:?}", network.recurrent_layer.state);

        // let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        // if new_cost > cost {
        //     println!("\nCOST REGRESSION\n");
        //     println!(
        //         "\nRECURRENT WEIGHTS: {:?}",
        //         network.recurrent_layer.recurrent_tree.all_layer_weights()
        //     );
        //     println!(
        //         "RECURRENT BIASES: {:?}",
        //         network.recurrent_layer.recurrent_tree.all_layer_biases()
        //     );
        //     println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
        //     println!("OUTPUT BIASES: {:?}", network.recurrent_layer.output_tree.biases);
        //     println!("FINAL STATE: {:?}", network.recurrent_layer.state);

        //     panic!();
        // }
    }

    println!(
        "\nRECURRENT WEIGHTS: {:?}",
        network.recurrent_layer.recurrent_tree.all_layer_weights()
    );
    println!(
        "RECURRENT BIASES: {:?}",
        network.recurrent_layer.recurrent_tree.all_layer_biases()
    );
    println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
    println!("OUTPUT BIASES: {:?}", network.recurrent_layer.output_tree.biases);
    println!("FINAL STATE: {:?}", network.recurrent_layer.state);

    assert!(cost < 0.001);
}

/// Output value seen 2 steps ago
#[test]
fn rnn_sanity_output_2_steps_back() {
    let input_size = 1;
    let output_size = 1;
    let state_size = 2;
    let learning_rate = 0.05;
    let mut network = build_test_network(input_size, output_size, state_size);

    fn gen_training_data() -> (Vec<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        let sequence_len = rand::thread_rng().gen_range(3usize, 4usize);
        let mut training_sequence = Vec::with_capacity(sequence_len);
        let mut expected_outputs = Vec::with_capacity(sequence_len);

        for i in 0..sequence_len {
            training_sequence.push(vec![rand::thread_rng().gen_range(-1., 1.)]);
            if i < 2 {
                expected_outputs.push(None);
            } else {
                expected_outputs.push(Some(training_sequence[i - 2].clone()));
            }
        }

        (training_sequence, expected_outputs)
    }

    let (training_sequence, expected_outputs) = gen_training_data();
    let (initial_total_cost, _output_gradients) =
        network.forward_propagate(&training_sequence, Some(&expected_outputs));
    println!("initial cost before training: {}", initial_total_cost);
    println!("initial outputs before training: {:?}", network.outputs);

    let mut last_iter_cost = initial_total_cost;
    for i in 0..5000 {
        let (training_sequence, expected_outputs) = gen_training_data();
        let new_cost = network.train_one_sequence(&training_sequence, &expected_outputs, learning_rate);
        if new_cost.is_nan() {
            panic!();
        }
        println!("\n[{}] cost: {}", i, new_cost);
        println!("[{}] inputs: {:?}", i, training_sequence.as_slice());
        println!("[{}] outputs: {:?}", i, &network.outputs[..training_sequence.len()]);
        println!(
            "[{}] expected: {:?}",
            i,
            expected_outputs
                .into_iter()
                .map(|o| o.unwrap_or_else(|| vec![-0.]))
                .collect::<Vec<_>>()
                .as_slice()
        );
        last_iter_cost = new_cost;
    }

    println!(
        "\nRECURRENT WEIGHTS: {:?}",
        network.recurrent_layer.recurrent_tree.all_layer_weights()
    );
    println!(
        "RECURRENT BIASES: {:?}",
        network.recurrent_layer.recurrent_tree.all_layer_biases()
    );
    println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
    println!("OUTPUT BIASES: {:?}", network.recurrent_layer.output_tree.biases);
    println!("FINAL STATE: {:?}", network.recurrent_layer.state);

    assert!(last_iter_cost < 0.001);
}
