/* eslint-disable no-restricted-globals */
/* global importScripts, ort */

// Import ONNX Runtime Web
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Configure ONNX runtime environment
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.backendHint = 'wasm';
ort.env.wasm.simd = false;
ort.env.wasm.numThreads = 1;

let session = null;
let currentBackend = 'unknown';

// Worker message handler
self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            console.log('[Worker] Loading model...');
            const startLoad = performance.now();

            // Try to load with WebGL first, fallback to WASM
            try {
                session = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['webgl', 'wasm']
                });
                currentBackend = 'webgl';
            } catch (webglError) {
                console.log('WebGL failed, falling back to WASM');
                session = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['wasm']
                });
                currentBackend = 'wasm';
            }

            const endLoad = performance.now();
            console.log(`[Worker] Model loaded in ${(endLoad - startLoad).toFixed(1)} ms`);
            console.log('[Worker] Using backend:', currentBackend);

            self.postMessage({ 
                type: 'loaded',
                backend: currentBackend
            });
        } catch (err) {
            self.postMessage({ 
                type: 'error', 
                message: err.message 
            });
        }
    }

    if (type === 'infer') {
        if (!session) {
            self.postMessage({ 
                type: 'error', 
                message: 'Model not loaded' 
            });
            return;
        }

        try {
            const startInfer = performance.now();
            const tensor = new ort.Tensor('float32', new Float32Array(tensorData), dims);
            const feeds = { images: tensor };
            const results = await session.run(feeds);
            const endInfer = performance.now();
            const inferTime = (endInfer - startInfer).toFixed(1);

            const outputKey = Object.keys(results)[0];
            const outputTensor = results[outputKey];

            self.postMessage(
                {
                    type: 'inference',
                    data: outputTensor.data.buffer,
                    dims: outputTensor.dims,
                    inferTime,
                    backend: currentBackend
                },
                [outputTensor.data.buffer]
            );
        } catch (err) {
            self.postMessage({ 
                type: 'error', 
                message: err.message 
            });
        }
    }
};