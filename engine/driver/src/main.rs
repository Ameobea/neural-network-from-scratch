use std::io::BufRead;

use libnn::*;
use rand::prelude::*;

fn main() {
    let mut init_weights = |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(-1.0..1.0) };

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
        let example_1 = rand::thread_rng().gen_range(-1.0..1.);
        let example_2 = rand::thread_rng().gen_range(-1.0..1.);
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
