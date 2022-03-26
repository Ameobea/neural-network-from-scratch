#![feature(
    box_syntax,
    label_break_value,
    try_blocks,
    const_maybe_uninit_assume_init,
    thread_local
)]

use layer_viz::{initialize_colorizer_luts, LayerVizState};
use libnn::{
    ActivationFunction, CostFunction, DenseLayer, Network, OutputLayer, Weight, AMEO, GAUSSIAN, GCU, IDENTITY,
    LEAKY_RELU, MEAN_SQUARED_ERROR, RELU, SIGMOID, SWISH, TANH,
};
use rand::prelude::*;
use wasm_bindgen::prelude::*;

mod layer_viz;

pub struct NNCtx {
    pub network: Network,
    pub viz_state: LayerVizState,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ActivationFunctionType {
    Identity = 0,
    Sigmoid = 1,
    Tanh = 2,
    ReLU = 3,
    LeakyReLU = 4,
    GCU = 5,
    Gaussian = 6,
    Swish = 7,
    Ameo = 8,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum CostFunctionType {
    MeanSquaredError = 0,
}

impl CostFunctionType {
    pub fn build(self) -> &'static dyn CostFunction {
        match self {
            CostFunctionType::MeanSquaredError => &MEAN_SQUARED_ERROR,
        }
    }
}

impl Into<&'static dyn ActivationFunction> for ActivationFunctionType {
    fn into(self) -> &'static dyn ActivationFunction {
        match self {
            ActivationFunctionType::Identity => &IDENTITY,
            ActivationFunctionType::Sigmoid => &SIGMOID,
            ActivationFunctionType::Tanh => &TANH,
            ActivationFunctionType::ReLU => &RELU,
            ActivationFunctionType::LeakyReLU => &LEAKY_RELU,
            ActivationFunctionType::GCU => &GCU,
            ActivationFunctionType::Gaussian => &GAUSSIAN,
            ActivationFunctionType::Swish => &SWISH,
            ActivationFunctionType::Ameo => &AMEO,
        }
    }
}

#[derive(Clone, Copy)]
pub enum InitWeightFnDefinition {
    Constant(Weight),
    ContinousUniformDistribution { min: Weight, max: Weight },
    // NormalDistribution { mean: Weight, std_deviation: Weight },
}

#[thread_local]
pub static mut RNG: pcg::Pcg = unsafe { std::mem::transmute((0u64, 0u64)) };

impl InitWeightFnDefinition {
    pub fn from_parts(fn_type: u8, arg0: Weight, arg1: Weight) -> Self {
        match fn_type {
            0 => InitWeightFnDefinition::Constant(arg0),
            1 => InitWeightFnDefinition::ContinousUniformDistribution { min: arg0, max: arg1 },
            // 2 => InitWeightFnDefinition::NormalDistribution { mean: arg0, std_deviation: arg1 },
            _ => panic!("Invalid init weights fn type"),
        }
    }

    pub fn build_weights(self) -> Box<dyn FnMut(usize, usize) -> Weight> {
        match self {
            InitWeightFnDefinition::Constant(val) => box move |_, _| val,
            InitWeightFnDefinition::ContinousUniformDistribution { min, max } =>
                box move |_, _| unsafe { &mut RNG }.gen_range(min, max),
        }
    }

    pub fn build_biases(self) -> Box<dyn FnMut(usize) -> Weight> {
        match self {
            InitWeightFnDefinition::Constant(val) => box move |_| val,
            InitWeightFnDefinition::ContinousUniformDistribution { min, max } =>
                box move |_| unsafe { &mut RNG }.gen_range(min, max),
        }
    }
}

#[derive(Clone, Copy)]
pub struct HiddenLayerDefinition {
    pub neuron_count: usize,
    pub activation_function_type: ActivationFunctionType,
    pub init_weights: InitWeightFnDefinition,
    pub init_biases: InitWeightFnDefinition,
}

impl HiddenLayerDefinition {
    pub const fn const_default() -> Self {
        HiddenLayerDefinition {
            neuron_count: 0,
            activation_function_type: ActivationFunctionType::Identity,
            init_weights: InitWeightFnDefinition::ContinousUniformDistribution { min: -1., max: 1. },
            init_biases: InitWeightFnDefinition::Constant(0.0),
        }
    }

    pub fn build_layer(&self, input_count: usize) -> DenseLayer {
        let mut init_weights = self.init_weights.build_weights();
        let mut init_biases = self.init_biases.build_biases();

        DenseLayer::new(
            self.neuron_count,
            input_count,
            &mut init_weights,
            &mut init_biases,
            self.activation_function_type.into(),
        )
    }
}

static mut HIDDEN_LAYER_DEFINITIONS: [HiddenLayerDefinition; 128] = [HiddenLayerDefinition::const_default(); 128];

#[wasm_bindgen]
pub fn store_hidden_layer_definition(
    index: usize,
    activation_function_type: u8,
    neuron_count: usize,
    init_weights_fn_type: u8,
    init_weights_fn_arg_0: Weight,
    init_weights_fn_arg_1: Weight,
    init_biases_fn_type: u8,
    init_biases_fn_arg_0: Weight,
    init_biases_fn_arg_1: Weight,
) {
    unsafe {
        HIDDEN_LAYER_DEFINITIONS[index] = HiddenLayerDefinition {
            neuron_count,
            activation_function_type: std::mem::transmute(activation_function_type),
            init_weights: InitWeightFnDefinition::from_parts(
                init_weights_fn_type,
                init_weights_fn_arg_0,
                init_weights_fn_arg_1,
            ),
            init_biases: InitWeightFnDefinition::from_parts(
                init_biases_fn_type,
                init_biases_fn_arg_0,
                init_biases_fn_arg_1,
            ),
        };
    }
}

static mut DID_INIT: bool = false;

fn maybe_init() {
    if unsafe { DID_INIT } {
        return;
    }
    unsafe { DID_INIT = true };

    console_error_panic_hook::set_once();
    unsafe { RNG = pcg::Pcg::seed_from_u64(10203040382934) };
    initialize_colorizer_luts();
}

#[wasm_bindgen]
pub fn create_nn_ctx(
    input_count: usize,
    output_count: usize,
    hidden_layer_count: usize,
    learning_rate: Weight,
    output_layer_activation_fn: u8,
    cost_fn_type: u8,
    output_layer_init_weights_fn_type: u8,
    output_layer_init_weights_fn_arg_0: Weight,
    output_layer_init_weights_fn_arg_1: Weight,
) -> *mut NNCtx {
    maybe_init();

    let mut hidden_layers: Vec<DenseLayer> = Vec::with_capacity(hidden_layer_count);

    let mut hidden_layer_input_count = input_count;
    for i in 0..hidden_layer_count {
        let def = unsafe { &HIDDEN_LAYER_DEFINITIONS[i] };
        hidden_layers.push(def.build_layer(hidden_layer_input_count));
        hidden_layer_input_count = def.neuron_count;
    }

    let output_layer_activation_fn_type: ActivationFunctionType =
        unsafe { std::mem::transmute(output_layer_activation_fn) };
    let cost_fn_type: CostFunctionType = unsafe { std::mem::transmute(cost_fn_type) };
    let mut init_output_layer_weights = InitWeightFnDefinition::from_parts(
        output_layer_init_weights_fn_type,
        output_layer_init_weights_fn_arg_0,
        output_layer_init_weights_fn_arg_1,
    )
    .build_weights();
    let output_layer = box OutputLayer::new(
        output_layer_activation_fn_type.into(),
        cost_fn_type.build(),
        &mut init_output_layer_weights,
        hidden_layers.last().unwrap().outputs.len(),
        output_count,
    );

    let network = Network {
        hidden_layers,
        outputs: output_layer,
        learning_rate,
    };
    let viz_state = LayerVizState::new(&network);

    let ctx = box NNCtx { network, viz_state };
    Box::into_raw(ctx)
}

#[wasm_bindgen]
pub fn free_nn_ctx(ctx: *mut NNCtx) { unsafe { drop(Box::from_raw(ctx)) } }

#[wasm_bindgen]
pub fn train(ctx: *mut NNCtx, example: &[Weight], expected: &[Weight], learning_rate: Weight) -> Weight {
    let network: &mut Network = unsafe { &mut (*ctx).network };

    network.train_one_example(example, expected, learning_rate)
}

#[wasm_bindgen]
pub fn train_many_examples(
    ctx: *mut NNCtx,
    examples: &[Weight],
    expected: &[Weight],
    learning_rate: Weight,
) -> Vec<Weight> {
    let network: &mut Network = unsafe { &mut (*ctx).network };

    let input_dims = network.hidden_layers[0].weights[0].len();
    let output_dims = network.outputs.outputs.len();
    let iterations = examples.len() / input_dims;
    let mut costs = Vec::with_capacity(iterations);

    assert_eq!(examples.len(), input_dims * iterations);
    assert_eq!(expected.len(), output_dims * iterations);

    for iteration in 0..iterations {
        let cost = network.train_one_example(
            &examples[iteration * input_dims..(iteration + 1) * input_dims],
            &expected[iteration * output_dims..(iteration + 1) * output_dims],
            learning_rate,
        );
        costs.push(cost);
    }

    costs
}

#[wasm_bindgen]
pub fn predict(ctx: *mut NNCtx, example: &[Weight]) -> Vec<Weight> {
    let network: &mut Network = unsafe { &mut (*ctx).network };
    network.compute(example).to_owned()
}

#[wasm_bindgen]
pub fn predict_batch(
    ctx: *mut NNCtx,
    mut example: Vec<Weight>,
    example_dim_to_replace: usize,
    min_input: Weight,
    max_input: Weight,
    steps: usize,
) -> Vec<Weight> {
    let network: &mut Network = unsafe { &mut (*ctx).network };
    let mut outputs: Vec<Weight> = Vec::with_capacity(steps * network.outputs.outputs.len());

    let range = max_input - min_input;
    let step_size = range / steps as f32;
    example[example_dim_to_replace] = min_input;
    for _ in 0..steps {
        outputs.extend_from_slice(network.compute(&example));
        example[example_dim_to_replace] += step_size;
    }

    outputs
}

#[wasm_bindgen]
pub fn update_viz(ctx: *mut NNCtx, example: &[Weight]) {
    let ctx = unsafe { &mut (*ctx) };
    ctx.network.forward_propagate(example);
    ctx.viz_state.update(&ctx.network);
}

#[wasm_bindgen]
pub fn get_viz_hidden_layer_colors(ctx: *mut NNCtx, hidden_layer_ix: usize) -> Vec<u8> {
    let ctx = unsafe { &mut (*ctx) };
    ctx.viz_state.hidden_layer_buffers[hidden_layer_ix].clone()
}

#[wasm_bindgen]
pub fn get_viz_output_layer_colors(ctx: *mut NNCtx) -> Vec<u8> {
    let ctx = unsafe { &mut (*ctx) };
    ctx.viz_state.output_layer_buffer.clone()
}
