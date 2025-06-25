importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.backendHint = 'webgl';
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 4;

let session = null;

self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            console.log('[Worker] Loading model...');
            const startLoad = performance.now();

            session = await ort.InferenceSession.create(modelUrl);

            const endLoad = performance.now();
            console.log(`[Worker] Model loaded in ${(endLoad - startLoad).toFixed(1)} ms`);

            console.log('[Worker] Backend:', ort.env.backendHint);

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
                },
                [outputTensor.data.buffer]
            );
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
    }
};
