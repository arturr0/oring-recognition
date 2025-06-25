importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Configure ONNX runtime for mobile compatibility
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.backendHint = 'auto'; // Let ORT choose the best backend
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 2;

let session = null;
let processing = false;

self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            console.log('[Worker] Loading model...');
            const startLoad = performance.now();

            // Added session options for mobile optimization
            const sessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true
            };

            session = await ort.InferenceSession.create(modelUrl, sessionOptions);

            const endLoad = performance.now();
            console.log(`[Worker] Model loaded in ${(endLoad - startLoad).toFixed(1)} ms`);
            console.log('[Worker] Backend:', ort.env.backendHint);

            self.postMessage({ type: 'loaded' });
        } catch (err) {
            console.error('[Worker] Load error:', err);
            self.postMessage({ type: 'error', message: err.message });
        }
    }

    if (type === 'infer') {
        if (!session) {
            self.postMessage({ type: 'error', message: 'Model not loaded' });
            return;
        }

        if (processing) {
            return; // Skip if already processing
        }

        processing = true;

        try {
            const startInfer = performance.now();

            // Create tensor directly from the input data to avoid extra copies
            const tensor = new ort.Tensor('float32', new Float32Array(tensorData), dims);
            const feeds = { [session.inputNames[0]]: tensor }; // Use dynamic input name

            const results = await session.run(feeds);
            const endInfer = performance.now();
            const inferTime = (endInfer - startInfer).toFixed(1);

            // Get the first output (works for most single-output models)
            const outputKey = session.outputNames[0];
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
            console.error('[Worker] Inference error:', err);
            self.postMessage({ type: 'error', message: err.message });
        } finally {
            processing = false;
        }
    }
};