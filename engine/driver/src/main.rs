use std::{io::BufRead, str::FromStr};

use clap::Command;
use libnn::{
    rnn::{RecurrentLayer, RecurrentNetwork, RecurrentTreeLayerDef},
    *,
};
use rand::prelude::*;

fn basic_network_demo() {
    let mut init_weights = |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(-1.0, 1.0) };

    let mut init_biases = |_neuron_ix| -> Weight { 0. };

    const INPUT_COUNT: usize = 2;
    const OUTPUT_COUNT: usize = 1;
    let learning_rate = 0.1;
    let hidden_layer_neuron_count = 10;

    let mut network: Network = Network {
        hidden_layers: vec![
            DenseLayer::new(
                hidden_layer_neuron_count,
                INPUT_COUNT,
                &mut init_weights,
                &mut init_biases,
                &Tanh,
            ),
            DenseLayer::new(
                hidden_layer_neuron_count,
                hidden_layer_neuron_count,
                &mut init_weights,
                &mut init_biases,
                &Tanh,
            ),
        ],
        outputs: Box::new(OutputLayer::new(
            &Identity,
            &MeanSquaredError,
            &mut init_weights,
            hidden_layer_neuron_count,
            OUTPUT_COUNT,
        )),
        learning_rate,
    };

    for _ in 0..2_000_000 {
        let example_1 = rand::thread_rng().gen_range(-1.0, 1.);
        let example_2 = rand::thread_rng().gen_range(-1.0, 1.);
        let expected_output = &[if example_1 > 0.5 || example_2 > example_1 {
            1.
        } else {
            0.
        }];

        network.train_one_example(&[example_1, example_2], expected_output, learning_rate);

        if network.outputs.costs[0] > 100_000. {
            println!(
                "hidden weight={:?}, hidden bias={:?}, output weight={:?}",
                network.hidden_layers[0].weights, network.hidden_layers[0].biases, network.outputs.weights
            );
            panic!("Cost fn explosion");
        }
    }

    println!(
        "hidden weight={:?}, hidden bias={:?}, output weight={:?}",
        network.hidden_layers[0].weights, network.hidden_layers[0].biases, network.outputs.weights
    );

    let stdin = std::io::stdin();
    loop {
        let line = stdin.lock().lines().next().unwrap().unwrap();
        if line.as_str() == "exit" {
            break;
        }
        let chunks: Vec<_> = line.split_ascii_whitespace().collect();
        if chunks.len() != 2 {
            continue;
        }
        let first: f32 = chunks[0].parse().unwrap();
        let second: f32 = chunks[1].parse().unwrap();

        println!("{:?}", network.compute(&[first, second]));
    }
}

fn rnn_2_ago(lookback: usize) {
    let output_size = 1;
    let input_size = 1;
    let state_size = 1;
    let learning_rate = 0.001;

    let init_recurrent_weights =
        |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(0., 0.1) };
    let init_recurrent_biases = |_output_ix: usize| -> Weight { 0. };
    let recurrent_activation_fn = &RELU;

    let mut init_output_weights =
        |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(0., 0.1) };
    let mut init_output_biases = |_output_ix: usize| -> Weight { 0. };
    let output_activation_fn = &RELU;

    let recurrent_layer_def = vec![
        RecurrentTreeLayerDef {
            input_count: input_size + state_size,
            output_count: state_size * 2,
            init_weights: Box::new(init_recurrent_weights),
            init_biases: Box::new(init_recurrent_biases),
            activation_fn: recurrent_activation_fn,
        },
        // RecurrentTreeLayerDef {
        //     input_count: state_size * 2,
        //     output_count: state_size * 2,
        //     init_weights: Box::new(init_recurrent_weights),
        //     init_biases: Box::new(init_recurrent_biases),
        //     activation_fn: recurrent_activation_fn,
        // },
        RecurrentTreeLayerDef {
            input_count: state_size * 2,
            output_count: state_size,
            init_weights: Box::new(init_recurrent_weights),
            init_biases: Box::new(init_recurrent_biases),
            activation_fn: recurrent_activation_fn,
        },
    ];

    let mut network = RecurrentNetwork {
        recurrent_layer: RecurrentLayer::new(
            4,
            input_size,
            recurrent_layer_def,
            &mut init_output_weights,
            &mut init_output_biases,
            output_activation_fn,
            state_size,
        ),
        output_layer: Box::new(OutputLayer::new(
            &RELU,
            &MEAN_SQUARED_ERROR,
            &mut |_, _| 1.,
            input_size,
            output_size,
        )),
        outputs: Vec::new(),
        recurrent_layer_outputs: Vec::new(),
    };

    fn gen_training_data(lookback: usize) -> (Vec<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        // let sequence_len = rand::thread_rng().gen_range(5usize, 5usize);
        let sequence_len = 5;
        let mut training_sequence = Vec::with_capacity(sequence_len);
        let mut expected_outputs = Vec::with_capacity(sequence_len);

        for i in 0..sequence_len {
            training_sequence.push(vec![rand::thread_rng().gen_range(0., 1.)]);
            if i < lookback {
                expected_outputs.push(None);
            } else {
                expected_outputs.push(Some(training_sequence[i - lookback].clone()));
            }
        }

        (training_sequence, expected_outputs)
    }

    for i in 0..100000 {
        let (training_sequence, expected_outputs) = gen_training_data(lookback);
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
    }

    println!(
        "\nRECURRENT WEIGHTS: {:?}",
        network.recurrent_layer.recurrent_tree.weights()
    );
    println!("OUTPUT WEIGHTS: {:?}", network.recurrent_layer.output_tree.weights);
    println!("FINAL STATE: {:?}\n", network.recurrent_layer.state);

    println!("\nDone training; ready to accept user input:\n");
    let stdin = std::io::stdin();
    loop {
        let line = stdin.lock().lines().next().unwrap().unwrap();
        if line.as_str() == "exit" {
            break;
        }
        let chunks = line.split_ascii_whitespace();
        let sequence = chunks
            .map(|v| Ok(vec![v.parse::<f32>()?]))
            .collect::<Result<Vec<Vec<f32>>, <f32 as FromStr>::Err>>();
        let sequence = match sequence {
            Ok(seq) => seq,
            Err(_) => {
                println!("Invalid number entered.");
                continue;
            },
        };

        println!("{:?}\n", network.predict(&sequence));
    }
}

pub fn main() {
    let mut app = Command::new("libnn driver")
        .subcommand(Command::new("basic"))
        .subcommand(Command::new("rnn-1-ago"))
        .subcommand(Command::new("rnn-2-ago"));
    let cli = app.clone().get_matches();

    match cli.subcommand() {
        Some(("basic", _)) => basic_network_demo(),
        Some(("rnn-1-ago", _)) => rnn_2_ago(1),
        Some(("rnn-2-ago", _)) => rnn_2_ago(2),
        _ => {
            let _ = app.print_help();
            return;
        },
    }
}
