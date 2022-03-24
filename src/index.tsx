import React from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';
import NoWasmSIMDMessage from './NoWasmSIMDMessage';
import { getSentry, initSentry } from './sentry';
import { getHasSIMDSupport } from './util';

initSentry();

const hasSIMDSupport = getHasSIMDSupport();

const toRender = hasSIMDSupport ? <App /> : <NoWasmSIMDMessage />;

createRoot(document.getElementById('root')!).render(toRender);

getSentry()?.captureMessage(
  hasSIMDSupport
    ? 'Wasm SIMD support detected'
    : 'Wasm SIMD support NOT detected; showing error banner'
);
