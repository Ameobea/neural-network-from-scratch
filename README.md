# Neural Network from Scratch in the Browser

![A screenshot of the neural network web application itself which shows the full UI, network response visualization, and cost plot](https://ameo.link/u/97k.png)

**Try it yourself**: <https://nn-viz.ameo.design>

## About

The goal of this project was to understand neural networks better by building them from the ground up.  I wanted to be able to see dynamically how choosing different network architectures + training parameters affects network performance and how well networks can various functions.

It allows users to define a neural network infrastructure by adding hidden layers, picking neuron counts + activation function types, and setting training parameters like learning rate.  The network then learns one of a variety of selectable target functions from which examples are randomly sampled to train it.

The "response" of the network over the entire range of possible inputs is then plotted as a 3D surface along with the target function to show how well the network has learned.

### Technical Details

The neural network implementation itself is built in Rust and compiled to WebAssembly using Wasm SIMD to accelerate training.

The training takes place on a dedicated thread by using a web worker and the excellent [Comlink](https://github.com/GoogleChromeLabs/comlink) library to communicate between the main/render thread and the training thread.

All the charts + visualizations are created using the excellent [echarts](https://echarts.apache.org/en/index.html) library.

The UI is created using [react-control-panel](https://github.com/ameobea/react-control-panel) which is a React port of the excellent [`control-panel`](https://github.com/freeman-lab/control-panel) library for easy GUI creation.

## Building + Developing

You'll need Rust nightly with WebAssembly support.  You can install Rust via easily via rustup: <https://rustup.rs/>

Then, add the latest nightly toolchain + switch to it:

`rustup default nightly`

Add WebAssembly support:

`rustup target add wasm32-unknown-unknown`

This project uses the [`just` command runner](https://github.com/casey/just) to simplify many tasks.  Install it with:

`cargo install just`.

You'll also need [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen):

`cargo install wasm-bindgen --version=0.2.74`

You'll need to install the `wasm-opt` tool from [`binaryen`](https://github.com/WebAssembly/binaryen).  You can download the executable from the Releases section on Github or build it yourself with CMake.

Then, you'll need tools for the web stack.  I use yarn for node package management, and you can either update the `Justfile` to change `yarn` to `npm` or install yarn 1.0 from here: <https://classic.yarnpkg.com/en/docs/install/>

That should be all you need!  To start the webpack dev server for hot-reloading and development, execute `just run` in the project root.

To create a release build, execute `just build` in the project root.  That will produce a fully functional static website in the `dist` directory.

## References

This is a list of some of the resources that I made use of while learning about neural networks and building this project:

* <https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/>
* <https://mlfromscratch.com/neural-networks-explained/#/>
* [MIT 6.S094: Recurrent Neural Networks for Steering Through Time](https://www.youtube.com/watch?v=nFTQ7kHQWtc&t=475s) <- Really helped me break through the wall of understanding some of the core concepts behind neural networks
