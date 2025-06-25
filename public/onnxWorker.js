importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Mobile-optimized configuration
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.backendHint = 'wasm'; // Force WASM for better mobile compatibility
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 2, 4); // Limit threads for mobile

let session = null;
let processing = false;

self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            console.log('[Worker] Loading model...');
            const startLoad = performance.now();

            // Mobile-optimized session options
            const sessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                executionMode: 'sequential', // Better for mobile
                enableProfiling: false
            };

            // Try to load with and without external data for mobile compatibility
            try {
                session = await ort.InferenceSession.create(modelUrl, sessionOptions);
            } catch (firstError) {
                console.warn('[Worker] First load attempt failed, trying fallback...');
                try {
                    sessionOptions.externalData = false;
                    session = await ort.InferenceSession.create(modelUrl, sessionOptions);
                } catch (secondError) {
                    throw new Error(`Failed to load model: ${firstError.message} and ${secondError.message}`);
                }
            }

            const endLoad = performance.now();
            console.log(`[Worker] Model loaded in ${(endLoad - startLoad).toFixed(1)} ms`);
            console.log('[Worker] Backend:', ort.env.backendHint);
            console.log('[Worker] Input names:', session.inputNames);
            console.log('[Worker] Output names:', session.outputNames);

            self.postMessage({ type: 'loaded' });
        } catch (err) {
            console.error('[Worker] Load error:', err);
            self.postMessage({ 
                type: 'error', 
                message: `Model load failed: ${err.message}` 
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

        if (processing) {
            return; // Skip if already processing
        }

        processing = true;

        try {
            const startInfer = performance.now();

            // Create tensor with mobile-optimized approach
            const tensor = new ort.Tensor('float32', new Float32Array(tensorData), dims);
            
            // Use dynamic input name from session
            const inputName = session.inputNames[0];
            const feeds = { [inputName]: tensor };

            const results = await session.run(feeds);
            const endInfer = performance.now();
            const inferTime = (endInfer - startInfer).toFixed(1);

            // Get first output
            const outputKey = session.outputNames[0];
            const outputTensor = results[outputKey];

            if (!outputTensor || !outputTensor.data) {
                throw new Error('Invalid output tensor received');
            }

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
            self.postMessage({ 
                type: 'error', 
                message: `Inference failed: ${err.message}` 
            });
        } finally {
            processing = false;
        }
    }
};