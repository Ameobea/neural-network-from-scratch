use std::mem::MaybeUninit;

use libnn::Network;
use palette::{
    encoding::{Linear, Srgb},
    rgb::Rgb,
    Gradient,
};

const COLORIZER_LUT_SIZE: usize = 512;
type ColorizerLUT = [[u8; 4]; COLORIZER_LUT_SIZE];

static mut COLORIZER_LUT: *const ColorizerLUT = std::ptr::null();

fn colorizer_lut() -> &'static ColorizerLUT { unsafe { &*COLORIZER_LUT } }

const COLORIZER_LUT_RANGE: [f32; 2] = [-2.5, 2.5];

fn build_lut(gradient: &Gradient<Rgb<Linear<Srgb>>>) -> ColorizerLUT {
    let [min, max] = COLORIZER_LUT_RANGE;
    let range = max - min;

    let mut lut = MaybeUninit::<ColorizerLUT>::uninit();
    for i in 0..COLORIZER_LUT_SIZE {
        // We want to be able to look up in the LUT by raw, unscaled values from [-2.5, 2.5]
        // The scaler converts the raw value to a value in [-1, 1]
        // Then we convert that to a value in [0, COLORIZER_LUT_SIZE]
        let x = min + (i as f32 / (COLORIZER_LUT_SIZE - 1) as f32) * range; // [-2.5, 2.5]
        let scaled_x = scale_output(x); // [-1, 1]
        let scaled_x = scaled_x + 1.0; // [0, 2]
        let scaled_x = scaled_x / 2.0; // [0, 1]

        let color = gradient.get(scaled_x);
        unsafe {
            (lut.as_mut_ptr() as *mut [u8; 4]).add(i).write([
                (color.red * 255.) as u8,
                (color.green * 255.) as u8,
                (color.blue * 255.) as u8,
                255,
            ])
        }
    }

    unsafe { lut.assume_init() }
}

fn build_even_color_steps(color_steps: &[[u8; 3]]) -> Vec<(f32, Rgb<Linear<Srgb>>)> {
    color_steps
        .into_iter()
        .enumerate()
        .map(|(i, [r, g, b])| {
            (
                i as f32 / ((color_steps.len() - 1) as f32),
                Rgb::new(*r as f32 / 255., *g as f32 / 255., *b as f32 / 255.),
            )
        })
        .collect()
}

pub fn initialize_colorizer_luts() {
    if unsafe { !COLORIZER_LUT.is_null() } {
        return;
    }

    let color_steps = build_even_color_steps(&[
        [10, 243, 255],
        [18, 194, 227],
        [27, 145, 198],
        [28, 99, 150],
        [22, 58, 83],
        [16, 16, 16],
        [112, 112, 10],
        [207, 207, 3],
        [255, 204, 0],
        [255, 102, 0],
        [255, 0, 0],
    ]);

    let lut = build_lut(&Gradient::with_domain(color_steps));
    unsafe { COLORIZER_LUT = Box::into_raw(box lut) };
}

/// Scales inputs into the range [-1, 1]
fn scale_output(value: f32) -> f32 { fastapprox::fast::tanh(value * 0.8) }

fn clamp(min: f32, max: f32, val: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

fn colorize_output(val: f32) -> [u8; 4] {
    let val = clamp(COLORIZER_LUT_RANGE[0], COLORIZER_LUT_RANGE[1], val);
    // Scale val from [-2.5, 2.5] to [0, COLORIZER_LUT_SIZE]
    let lut_ix = ((val + 2.5) * (COLORIZER_LUT_SIZE - 1) as f32 / 5.) as usize;
    debug_assert!(lut_ix < COLORIZER_LUT_SIZE);

    let lut = colorizer_lut();
    unsafe { *lut.get_unchecked(lut_ix) }
}

const VIZ_SCALE_MULTIPLIER: usize = 16;

pub fn build_layer_outputs_buf(outputs: &[f32]) -> Vec<u8> {
    vec![0; outputs.len() * VIZ_SCALE_MULTIPLIER * VIZ_SCALE_MULTIPLIER * 4]
}

pub struct LayerVizState {
    pub hidden_layer_buffers: Vec<Vec<u8>>,
    pub output_layer_buffer: Vec<u8>,
}

impl LayerVizState {
    pub fn new(network: &Network) -> Self {
        let hidden_layer_buffers = network
            .hidden_layers
            .iter()
            .map(|layer| build_layer_outputs_buf(&layer.outputs))
            .collect();
        let output_layer_buffer = build_layer_outputs_buf(&network.outputs.outputs);

        Self {
            hidden_layer_buffers,
            output_layer_buffer,
        }
    }

    fn populate_layer_outputs_buf(buf: &mut [u8], outputs: &[f32]) {
        let buf_ptr = buf.as_mut_ptr() as *mut u8 as *mut u32;

        for (i, output) in outputs.iter().enumerate() {
            let color = colorize_output(*output);
            if color[0] == 0 && color[1] == 0 && color[2] == 0 {
                panic!("{}", *output);
            }
            let color: u32 = unsafe { std::mem::transmute(color) };

            for y in 0..VIZ_SCALE_MULTIPLIER {
                for x in 0..VIZ_SCALE_MULTIPLIER {
                    unsafe {
                        std::ptr::write(
                            buf_ptr.add((outputs.len() * y * VIZ_SCALE_MULTIPLIER) + i * VIZ_SCALE_MULTIPLIER + x),
                            color,
                        )
                    }
                }
            }
        }
    }

    pub fn update(&mut self, network: &Network) {
        for (layer_ix, hidden_layer) in network.hidden_layers.iter().enumerate() {
            Self::populate_layer_outputs_buf(&mut self.hidden_layer_buffers[layer_ix], &hidden_layer.outputs);
        }
        Self::populate_layer_outputs_buf(&mut self.output_layer_buffer, &network.outputs.outputs);
    }
}
