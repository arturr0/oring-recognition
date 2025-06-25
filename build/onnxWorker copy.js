/* eslint-disable no-restricted-globals */
/* eslint-env worker */
/* global ort */

// Load ONNX Runtime Web from CDN
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Set wasm paths
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let session = null;

self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            session = await ort.InferenceSession.create(modelUrl);
            self.postMessage({ type: 'loaded' });
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
    }

    if (type === 'infer') {
        if (!session) {
            self.postMessage({ type: 'error', message: 'Model not loaded' });
            return;
        }
        try {
            const tensor = new ort.Tensor('float32', new Float32Array(tensorData), dims);
            // IMPORTANT: Use your model's actual input name here, error says it's "images"
            const feeds = { images: tensor };

            const results = await session.run(feeds);

            // Assuming the output key is "output" - change if different
            const outputKey = Object.keys(results)[0]; // safer to get first key dynamically
            const outputTensor = results[outputKey];

            self.postMessage({
                type: 'inference',
                data: outputTensor.data.buffer,
                dims: outputTensor.dims
            }, [outputTensor.data.buffer]);
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
    }
};
