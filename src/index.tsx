import React from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';
import { getSentry, initSentry } from './sentry';
import { getHasSIMDSupport } from './util';

initSentry();

const hasSIMDSupport = getHasSIMDSupport();

createRoot(document.getElementById('root')!).render(<App />);

getSentry()?.captureMessage(
  hasSIMDSupport
    ? 'Wasm SIMD support detected'
    : 'Wasm SIMD support NOT detected; falling back to non-SIMD code'
);
