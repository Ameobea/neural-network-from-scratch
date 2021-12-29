import React from 'react';

import './NoWasmSIMDMessage.css';

const NoWasmSIMDMessage: React.FC = () => (
  <div className='no-wasm-simd-support-message'>
    <h1>Your browser does not support WebAssembly SIMD</h1>
    <p>
      Wasm SIMD is required to use this tool. It provides the ability to greaty accelerate the
      neural network&apos;s training by performing multiple arithmetic operations in the same
      instruction.
    </p>
    <p>Please use a modern browser that supports Wasm SIMD such as Chrome or Firefox.</p>
  </div>
);

export default NoWasmSIMDMessage;
