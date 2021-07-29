use rand::prelude::*;

use super::*;

#[test]
fn test_dense_layer_forward_propagation() {
    let inputs = &[1.2, -2.0];

    let mut dense_layer = DenseLayer {
        weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
        biases: vec![1.0, -2.0],
        neuron_gradients: vec![0.; 2],
        activation_fn: &Sigmoid,
        outputs_before_activation: vec![0., 0.],
        outputs: vec![0., 0.],
    };

    let sigmoid = Sigmoid;
    dense_layer.forward_propagate(inputs);
    let expected_outputs = vec![
        sigmoid.get_output(1.2 * -1.2 + -2. * 0.4 + 1.),
        sigmoid.get_output(1.2 * 2. + -2. * -1. + -2.),
    ];
    assert_eq!(expected_outputs, dense_layer.outputs);
}

#[test]
fn test_output_layer_forward_propagation() {
    let inputs = &[1.2, -2.0];

    let mut output_layer = OutputLayer {
        weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
        neuron_gradients: vec![0.; 2],
        activation_fn: &Sigmoid,
        outputs_before_activation: vec![0., 0.],
        outputs: vec![0., 0.],
        errors: vec![0., 0.],
        costs: vec![0., 0.],
        cost_fn: &MeanSquaredError,
    };

    let sigmoid = Sigmoid;
    output_layer.forward_propagate(inputs);
    let expected_outputs = vec![
        sigmoid.get_output(1.2 * -1.2 + -2. * 0.4),
        sigmoid.get_output(1.2 * 2. + -2. * -1.),
    ];
    assert_eq!(expected_outputs, output_layer.outputs);
}

#[test]
fn test_forward_propagation() {
    let mut network: Network = Network {
        hidden_layers: vec![DenseLayer {
            weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
            biases: vec![1.0, -2.0],
            neuron_gradients: vec![0.; 2],
            activation_fn: &Sigmoid,
            outputs_before_activation: vec![0., 0.],
            outputs: vec![0., 0.],
        }],
        outputs: Box::new(OutputLayer {
            weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
            neuron_gradients: vec![0.; 2],
            activation_fn: &Sigmoid,
            outputs_before_activation: vec![0., 0.],
            outputs: vec![0., 0.],
            errors: vec![0., 0.],
            costs: vec![0., 0.],
            cost_fn: &MeanSquaredError,
        }),
        learning_rate: 0.2,
    };

    let inputs = &[1.2, -2.0];
    network.forward_propagate(inputs);

    let sigmoid = &Sigmoid;
    let a = sigmoid.get_output(1.2 * -1.2 + -2. * 0.4 + 1.);
    let b = sigmoid.get_output(1.2 * 2. + -2. * -1. + -2.);
    let expected_outputs = vec![
        sigmoid.get_output(a * -1.2 + b * 0.4),
        sigmoid.get_output(a * 2. + b * -1.),
    ];
    assert_eq!(expected_outputs, network.outputs.outputs);
}

#[test]
fn test_error_computation() {
    let mut output_layer = OutputLayer {
        weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
        neuron_gradients: vec![0.; 2],
        activation_fn: &Sigmoid,
        outputs_before_activation: vec![0., 0.],
        outputs: vec![-0.2, 2.4],
        errors: vec![0., 0.],
        costs: vec![0., 0.],
        cost_fn: &MeanSquaredError,
    };

    let actual_values = &[0.0, 1.0];
    let expected_errors = &[0.0 - -0.2, 1.0 - 2.4];
    let expected_costs = [
        (expected_errors[0] * expected_errors[0]),
        (expected_errors[1] * expected_errors[1]),
    ];
    output_layer.compute_costs(actual_values);
    assert_eq!(output_layer.errors, *expected_errors);
    assert_eq!(output_layer.costs, expected_costs);
}

#[test]
fn test_single_neuron_weight_updating() {
    let inputs = &[0.4, -0.3];
    let desired_outputs = &[0.];

    let mut output_layer = OutputLayer {
        weights: vec![vec![-0.2, 0.9]],
        neuron_gradients: vec![0.],
        activation_fn: &Sigmoid,
        outputs: vec![0.0],
        outputs_before_activation: vec![0.],
        errors: vec![0.],
        costs: vec![0.],
        cost_fn: &MeanSquaredError,
    };

    // Run forward once with initial random weights and compute our costs
    output_layer.forward_propagate(inputs);
    output_layer.compute_costs(desired_outputs);
    let mut before_costs = output_layer.costs.clone();
    println!("INPUTS: {:?}", inputs);
    println!("DESIRED OUTPUTS: {:?}\n", desired_outputs);
    println!("INITIAL weights: {:?}", output_layer.weights);
    println!(
        "INITIAL pre-activation outputs: {:?}",
        output_layer.outputs_before_activation
    );
    println!("INITIAL post-activation outputs: {:?}", output_layer.outputs);
    println!("INITIAL costs: {:?}\n", before_costs);

    for _ in 0..500 {
        before_costs = output_layer.costs.clone();

        output_layer.compute_gradients();
        println!("Gradients: {:?}", output_layer.neuron_gradients);
        output_layer.update_weights(inputs, 0.5);
        println!("AFTER weights: {:?}", output_layer.weights);

        output_layer.forward_propagate(inputs);
        output_layer.compute_costs(desired_outputs);
        let after_costs = output_layer.costs.clone();
        // Updating these weights should have reduced the costs; that's the whole point of doing it.
        println!(
            "AFTER pre-activation outputs: {:?}",
            output_layer.outputs_before_activation
        );
        println!("AFTER post-activation outputs : {:?}", output_layer.outputs);
        println!("Before costs={:?}", before_costs,);
        println!("AFTER costs= {:?}", after_costs);
        assert!(before_costs[0] >= output_layer.costs[0]);
    }
}

#[test]
fn test_weight_updating() {
    let mut output_layer = OutputLayer {
        weights: vec![vec![-1.2, 0.4], vec![2.0, -1.0]],
        neuron_gradients: vec![0.; 2],
        activation_fn: &Sigmoid,
        outputs: vec![-0.2, 2.4],
        outputs_before_activation: vec![0., 0.],
        errors: vec![0., 0.],
        costs: vec![0., 0.],
        cost_fn: &MeanSquaredError,
    };

    // Run forward once with initial random weights and compute our costs
    let inputs = &[0.2, -0.8];
    let desired_outputs = &[0.0, 1.0];
    output_layer.forward_propagate(inputs);
    output_layer.compute_costs(desired_outputs);
    let mut before_costs;
    println!("Initial outputs: {:?}", output_layer.outputs);

    // Run one iteration of what equates to training, computing gradients and updating weights in
    // order to minimize costs
    for _ in 0..100 {
        before_costs = output_layer.costs.clone();

        output_layer.compute_gradients();
        println!("Gradients: {:?}", output_layer.neuron_gradients);
        output_layer.update_weights(inputs, 0.5);
        println!("Updated weights: {:?}", output_layer.weights);

        output_layer.forward_propagate(inputs);
        output_layer.compute_costs(desired_outputs);
        let after_costs = output_layer.costs.clone();
        // Updating these weights should have reduced the costs; that's the whole point of doing it.
        println!("After outputs: {:?}", output_layer.outputs);
        println!("Before costs={:?}", before_costs,);
        println!("After costs= {:?}", after_costs);
        assert!(before_costs[0] > output_layer.costs[0]);
        assert!(before_costs[1] > output_layer.costs[1]);
    }
}

#[test]
fn test_hidden_layer_single_weight_updating() {
    let mut dense_layer = DenseLayer {
        weights: vec![vec![1.0]],
        biases: vec![0.0],
        neuron_gradients: vec![0.],
        activation_fn: &Identity,
        outputs_before_activation: vec![0.],
        outputs: vec![0.],
    };

    // Run forward once with initial random weights and compute our costs
    let inputs = &[1.];
    let _outputs = &[0.];
    dense_layer.forward_propagate(inputs);

    let output_weights = &[vec![1.]];
    // Gradient is calculated for an output layer with an identity activation function and an
    // expected output of 0 which yields an error of -1 and a gradient of -2.
    let fake_output_gradients = &[-2.];

    dense_layer.compute_gradients(output_weights, fake_output_gradients);
    println!("Gradients: {:?}", dense_layer.neuron_gradients);
    dense_layer.update_weights(inputs, 0.5);
    println!("Updated weights: {:?}", dense_layer.weights);

    // SO:
    //
    // The weight we're connected to on the output layer has a value of 1.0 which means whatever we
    // output from our activation function (the identity function) will positively impact the
    // value of the output layer.  The error is negative which means the gradient is
    // negative for the output neuron.  That means that we need to move in the opposite direction of
    // where we're currently pointing. Our weight is positive and our input is positive so we're
    // pointing up.  Moving along the output neuron's gradient will be achieved by reducing our
    // weight.
    //
    // Reducing our weight means that the value we output will be less since the input connected to
    // the weight contributes positively to our output.  We will contribute less to the error,
    // causing us to move along the output neuron's gradient and reduce its error.

    assert!(dense_layer.weights[0][0] < 1.);
}

#[test]
fn test_hidden_layer_single_neuron_bias_updating() {
    let mut dense_layer = DenseLayer {
        weights: vec![vec![1.0]],
        biases: vec![0.0],
        neuron_gradients: vec![0.],
        activation_fn: &Identity,
        outputs_before_activation: vec![0.],
        outputs: vec![0.],
    };

    // Run forward once with initial random weights and compute our costs
    let inputs = &[1.];
    let _outputs = &[0.];
    dense_layer.forward_propagate(inputs);

    let output_weights = &[vec![1.]];
    // Gradient is calculated for an output layer with an identity activation function and an
    // expected output of 0 which yields an error of -1 and a gradient of -2.
    let fake_output_gradients = &[-2.];

    dense_layer.compute_gradients(output_weights, fake_output_gradients);
    println!("Gradients: {:?}", dense_layer.neuron_gradients);
    dense_layer.update_biases(0.5);
    println!("Updated biases: {:?}", dense_layer.biases);

    // We leave weights constant, so weights are still positively impacting the output of this
    // neuron which is positively impacting the output of the output neuron which is creating
    // error there.
    //
    // Bias should be updated downwards which will subtract from the output of the hidden neuron,
    // reducing our impact on the output of the output neuron.
    assert!(dense_layer.biases[0] < 0.);
}

#[test]
fn test_most_basic_full_neural_net_training() {
    // Create the simplest possible "neural network".  Single input, single hidden layer, single
    // output.  Identity for all activation functions.
    //
    // We initialize all weights to 1 and biases to 0, with the goal of the network being to learn
    // how to `-1 * input`.

    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = 1;
    let hidden_layer_neuron_count = 1;
    // We use a miniscule learning rate due to the huge input values.
    let learning_rate = 0.005;
    let mut network: Network = Network {
        hidden_layers: vec![DenseLayer::new(
            hidden_layer_neuron_count,
            INPUT_COUNT,
            &mut |_, _| 1.,
            &mut |_| 0.,
            &Identity,
        )],
        outputs: Box::new(OutputLayer::new(
            &Identity,
            &MeanSquaredError,
            &mut |_, _| 1.,
            hidden_layer_neuron_count,
            OUTPUT_COUNT,
        )),
        learning_rate,
    };

    let input = 5.;
    let training_output = -5.;

    // Train the network on the example one time.
    network.forward_propagate(&[input]);
    network.outputs.compute_costs(&[training_output]);
    let mut start_cost = network.outputs.costs[0];
    println!("BEFORE COST: {}", start_cost);
    network.outputs.compute_gradients();
    // The initial output will be input * hidden_layer_weight * output_layer_weight = 5 * 1 * 1 = 5.
    println!("dense layer outputs: {}", network.hidden_layers[0].outputs[0]);
    assert_eq!(network.outputs.outputs_before_activation[0], 5.);
    assert_eq!(network.outputs.outputs[0], 5.);
    // The error is going to be -10.  The gradient of the output layer will be 2 * -10 * 1 = -20.
    assert_eq!(network.outputs.errors[0], -10.);
    assert_eq!(network.outputs.neuron_gradients[0], -20.);

    // Compute gradients for the hidden layer
    network.hidden_layers[0].compute_gradients(
        network.outputs.weights.as_slice(),
        network.outputs.neuron_gradients.as_slice(),
    );

    // Actually update output layer weights using the computed gradient and output from the hidden
    // layer.
    network.outputs.update_weights(&[input], learning_rate);
    // The input to the output layer's weight is 5 since hidden layer weight is 1.  This positively
    // contributes to the outuput of the output layer and to move along the negative gradient,
    // we will reduce the weight by (-20 * 1) * 0.5 * 5 = -50.  New output layer weight is now
    // -49. assert_eq!(network.outputs.weights[0][0], -49.);

    // Now we update the weights for the hidden layer
    network.hidden_layers[0].update_weights(&[input], learning_rate);
    // Our input is positive.  The gradient of the output neuron is negative meaning that we need to
    // move in the opposite direction.
    //
    // The action to take to achieve that is to reduce the hidden neuron's weight which will reduce
    // our positive impact on the output and move us along the output gradient.
    assert!(network.hidden_layers[0].weights[0][0] < 1.);

    // Update biases as well.  The bias will go down too since it contributes positively to the
    // output of the hidden layer and we want to move in the opposite direction of that to
    // minimize the output error which is correlated positively to the output layer weight.
    network.hidden_layers[0].update_biases(learning_rate);
    assert!(network.hidden_layers[0].biases[0] < 0.);
    // The bias should be greater than the weight since the weight is updated further due to being
    // multiplied by the input which is 5 assert!(network.hidden_layers[0].biases[0] >
    // network.hidden_layers[0].weights[0][0]);

    // Now, we re-run the same training example and re-compute costs.  The costs should be lower
    // since we've just updated the network to better fit that same example.
    let mut end_cost = 0.;
    while start_cost > end_cost {
        network.forward_propagate(&[input]);
        network.outputs.compute_costs(&[training_output]);
        end_cost = network.outputs.costs[0];
        println!("AFTER COST: {}", end_cost);
        assert!(end_cost < start_cost);
        start_cost = end_cost;

        network.train_one_example(&[input], &[training_output], learning_rate);
        network.forward_propagate(&[input]);
        network.outputs.compute_costs(&[training_output]);
        end_cost = network.outputs.costs[0];
        println!("AFTER COST: {}", end_cost);
        assert!(end_cost <= start_cost);
    }

    println!(
        "hidden weight={}, hidden bias={}, output weight={}",
        network.hidden_layers[0].weights[0][0], network.hidden_layers[0].biases[0], network.outputs.weights[0][0]
    );

    // Check that we've successfully learned!!
    network.forward_propagate(&[input]);
    assert!((network.outputs.outputs[0] + 5.).abs() < 0.00001);
}

#[test]
fn test_learns_to_always_output_1() {
    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = 1;
    let learning_rate = 0.01;

    let mut network: Network = Network {
        hidden_layers: vec![DenseLayer::new(
            1,
            INPUT_COUNT,
            &mut |_, _| rand::thread_rng().gen_range(-1.0..1.0),
            &mut |_| 0.,
            &Identity,
        )],
        outputs: Box::new(OutputLayer::new(
            &Identity,
            &MeanSquaredError,
            &mut |_, _| thread_rng().gen_range(-1.0..1.),
            1,
            OUTPUT_COUNT,
        )),
        learning_rate,
    };

    // Train it to always output 1.  Network will learn to set a hidden layer weight of 0 and pick a
    // bias and output weight that when multiplied together yield very close to 1.
    let mut rng = rand::thread_rng();
    for _ in 0..100_000 {
        let example = rng.gen_range(-1.0..1.0);
        network.train_one_example(&[example], &[1.], learning_rate);

        if network.hidden_layers[0].weights[0][0].is_nan() {
            panic!();
        }

        let cost = network.outputs.costs[0];
        println!("Cost: {}", cost);

        if cost > 100_000. {
            println!(
                "hidden weight={}, hidden bias={}, output weight={}",
                network.hidden_layers[0].weights[0][0],
                network.hidden_layers[0].biases[0],
                network.outputs.weights[0][0]
            );
            panic!("Cost fn explosion");
        }
    }

    println!(
        "hidden weight={}, hidden bias={}, output weight={}",
        network.hidden_layers[0].weights[0][0], network.hidden_layers[0].biases[0], network.outputs.weights[0][0]
    );

    let cost = network.outputs.costs[0];
    assert!(cost < 0.0001);
}

#[test]
fn test_multiplies_inputs() {
    let mut init_weights = |_output_ix: usize, _input_ix: usize| -> Weight { rand::thread_rng().gen_range(-0.2..0.2) };

    let mut init_biases = |_neuron_ix| -> Weight { 0. };

    const INPUT_COUNT: usize = 2;
    const OUTPUT_COUNT: usize = 1;
    let learning_rate = 0.5;
    let hidden_layer_neuron_count = 8;

    let mut network: Network = Network {
        hidden_layers: vec![
            DenseLayer::new(
                hidden_layer_neuron_count,
                INPUT_COUNT,
                &mut init_weights,
                &mut init_biases,
                &Sigmoid,
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

    for _ in 0..100_000 {
        let example_1 = rand::thread_rng().gen_range(0.0..1.);
        let example_2 = rand::thread_rng().gen_range(0.0..1.);
        let expected_output = &[example_1 * example_2];

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

    let inputs = &[0.5, 0.0];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!(network.outputs.outputs[0] < 0.01);
    let inputs = &[1.0, 1.0];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!(network.outputs.outputs[0] > 0.95);
    let inputs = &[0.5, 0.5];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!((network.outputs.outputs[0] - 0.25).abs() < 0.01);
    let inputs = &[1.0, 0.2];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!((network.outputs.outputs[0] - 0.2).abs() < 0.01);
    let inputs = &[0.0, 0.0];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!((network.outputs.outputs[0]).abs() < 0.1);
    let inputs = &[0.8, 0.8];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!((network.outputs.outputs[0] - 0.64).abs() < 0.01);
    let inputs = &[0.9, 0.9];
    println!("inputs={:?}, outputs={:?}", inputs, network.compute(inputs));
    assert!((network.outputs.outputs[0] - 0.81).abs() < 0.01);
}

#[test]
fn test_complex_network_application() {
    const INPUT_COUNT: usize = 2;
    const OUTPUT_COUNT: usize = 1;
    let learning_rate = 1.;
    let hidden_layer_neuron_count = 1;

    let mut init_weights = |_, _| 0.;
    let mut init_biases = |_| 0.;

    let mut network: Network = Network {
        hidden_layers: vec![
            DenseLayer::new(
                hidden_layer_neuron_count,
                INPUT_COUNT,
                &mut init_weights,
                &mut init_biases,
                &Sigmoid,
            ),
            DenseLayer::new(
                hidden_layer_neuron_count,
                hidden_layer_neuron_count,
                &mut |_, _| 1.,
                &mut init_biases,
                &ReLU,
            ),
        ],
        outputs: Box::new(OutputLayer::new(
            &Tanh,
            &MeanSquaredError,
            &mut |_, _| -2.,
            hidden_layer_neuron_count,
            OUTPUT_COUNT,
        )),
        learning_rate,
    };

    let inputs = [1., 0.];
    let _outputs = network.compute(&inputs);

    assert_eq!(network.hidden_layers[0].outputs_before_activation[0], 0.);
    assert_eq!(network.hidden_layers[0].outputs[0], 0.5);

    assert_eq!(network.hidden_layers[1].outputs_before_activation[0], 0.5);
    assert_eq!(network.hidden_layers[1].outputs[0], 0.5);

    assert_eq!(network.outputs.outputs_before_activation[0], -2. * 0.5);
    assert_eq!(network.outputs.outputs[0], (-1.0f32).tanh());
}
