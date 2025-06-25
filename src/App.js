import React, { useEffect, useRef, useState } from 'react';

const MODEL_INPUT_SIZE = 640;
const classNames = ['BLOCK', 'INNER', 'OK', 'OUTER', 'SCAR', 'TEAR'];
const NMS_THRESHOLD = 0.5;
const CONFIDENCE_THRESHOLD = 0.4;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const isProcessingRef = useRef(false);

  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [referenceSize, setReferenceSize] = useState(10);
  const [calibrationComplete, setCalibrationComplete] = useState(false);
  const [pixelsPerMM, setPixelsPerMM] = useState(null);
  const [selectedReferenceBox, setSelectedReferenceBox] = useState(null);

  useEffect(() => {
    const constraints = {
      audio: false,
      video: {
        facingMode: 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 },
      }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      })
      .catch(e => {
        setError('Camera error: ' + e.message);
        setStatus('error');
      });
  }, []);

  useEffect(() => {
    workerRef.current = new Worker('onnxWorker.js');
    const worker = workerRef.current;

    worker.onmessage = e => {
      const { type } = e.data;

      if (type === 'loaded') {
        setStatus('ready');
      } else if (type === 'inference') {
        const data = new Float32Array(e.data.data);
        const dims = e.data.dims;
        const parsedBoxes = parseYOLOv5Output(data, dims);
        const nmsBoxes = nonMaxSuppression(parsedBoxes);
        setBoxes(nmsBoxes);
        isProcessingRef.current = false;
      } else if (type === 'error') {
        setError('Inference error: ' + e.data.message);
        setStatus('error');
        isProcessingRef.current = false;
      }
    };

    worker.postMessage({ type: 'loadModel', modelUrl: '/best.onnx' });
    return () => worker.terminate();
  }, []);

  function preprocess(frame) {
    const offscreen = document.createElement('canvas');
    offscreen.width = MODEL_INPUT_SIZE;
    offscreen.height = MODEL_INPUT_SIZE;
    const ctx = offscreen.getContext('2d');
    ctx.drawImage(frame, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const imgData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;

    const data = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    for (let i = 0; i < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; i++) {
      data[i] = imgData[i * 4] / 255;
      data[i + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i * 4 + 1] / 255;
      data[i + 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i * 4 + 2] / 255;
    }

    return data.buffer;
  }

  function parseYOLOv5Output(data, dims) {
    const [batch, num_boxes, num_attrs] = dims;
    const boxes = [];

    for (let i = 0; i < num_boxes; i++) {
      const offset = i * num_attrs;
      const slice = data.subarray(offset, offset + num_attrs);
      const [x, y, w, h, conf, ...classConfs] = slice;
      const classId = classConfs.indexOf(Math.max(...classConfs));
      const totalConf = conf * classConfs[classId];

      if (totalConf > CONFIDENCE_THRESHOLD) {
        boxes.push({
          x1: x - w / 2,
          y1: y - h / 2,
          x2: x + w / 2,
          y2: y + h / 2,
          width: w,
          height: h,
          label: classNames[classId] || 'unknown',
          confidence: totalConf,
          classId
        });
      }
    }

    return boxes;
  }

  function nonMaxSuppression(boxes) {
    const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selected = [];

    while (sortedBoxes.length) {
      const current = sortedBoxes.shift();
      selected.push(current);
      for (let i = sortedBoxes.length - 1; i >= 0; i--) {
        if (sortedBoxes[i].classId === current.classId) {
          const iou = calculateIoU(current, sortedBoxes[i]);
          if (iou > NMS_THRESHOLD) sortedBoxes.splice(i, 1);
        }
      }
    }

    return selected;
  }

  function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    return inter / (area1 + area2 - inter);
  }

  function calculatePhysicalSize(box, ppm) {
    const widthPx = box.width * MODEL_INPUT_SIZE;
    const heightPx = box.height * MODEL_INPUT_SIZE;
    return (widthPx + heightPx) / 2 / ppm;
  }

  function startCalibration() {
    if (referenceSize > 0 && !isNaN(referenceSize)) {
      setCalibrationMode(true);
      setError(null);
    } else {
      setError('Enter valid reference size.');
    }
  }

  function completeCalibration() {
    if (selectedReferenceBox && referenceSize > 0 && !isNaN(referenceSize)) {
      const px = (selectedReferenceBox.width + selectedReferenceBox.height) / 2 * MODEL_INPUT_SIZE;
      setPixelsPerMM(px / referenceSize);
      setCalibrationComplete(true);
      setCalibrationMode(false);
    } else {
      setError('Select reference box and size.');
    }
  }

  function resetCalibration() {
    setCalibrationComplete(false);
    setPixelsPerMM(null);
    setSelectedReferenceBox(null);
    setCalibrationMode(false);
  }

  function handleCanvasClick(e) {
    if (!calibrationMode) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;

    const matchedBox = boxes.find(box => {
      const bx = (box.x1 / MODEL_INPUT_SIZE) * rect.width;
      const by = (box.y1 / MODEL_INPUT_SIZE) * rect.height;
      const bw = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * rect.width;
      const bh = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * rect.height;
      return x >= bx && x <= bx + bw && y >= by && y <= by + bh;
    });

    if (matchedBox) setSelectedReferenceBox(matchedBox);
  }

  useEffect(() => {
    if (status !== 'ready') return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const draw = () => {
      if (video.readyState >= 2) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        canvas.width = vw;
        canvas.height = vh;
        ctx.drawImage(video, 0, 0, vw, vh);

        boxes.forEach(box => {
          const x = box.x1 * vw / MODEL_INPUT_SIZE;
          const y = box.y1 * vh / MODEL_INPUT_SIZE;
          const w = (box.x2 - box.x1) * vw / MODEL_INPUT_SIZE;
          const h = (box.y2 - box.y1) * vh / MODEL_INPUT_SIZE;

          ctx.strokeStyle = selectedReferenceBox === box ? 'yellow' : (box.label === 'OK' ? 'lime' : 'red');
          ctx.lineWidth = selectedReferenceBox === box ? 4 : 2;
          ctx.strokeRect(x, y, w, h);

          let label = `${box.label} (${(box.confidence * 100).toFixed(1)}%)`;
          if (box.label === 'OK' && pixelsPerMM) {
            const sizeMM = calculatePhysicalSize(box, pixelsPerMM);
            label += ` - Ø${sizeMM.toFixed(1)}mm`;
          }

          ctx.fillStyle = 'white';
          ctx.fillRect(x - 2, y - 18, ctx.measureText(label).width + 4, 18);
          ctx.fillStyle = 'black';
          ctx.fillText(label, x, y - 4);
        });

        if (!isProcessingRef.current) {
          isProcessingRef.current = true;
          const tensorData = preprocess(video);
          workerRef.current.postMessage({
            type: 'infer',
            tensorData,
            dims: [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
          }, [tensorData]);
        }
      }

      requestAnimationFrame(draw);
    };

    requestAnimationFrame(draw);
  }, [status, boxes, calibrationMode, selectedReferenceBox, pixelsPerMM]);

  return (
    <div style={{ padding: 10, maxWidth: 1000, margin: '0 auto', textAlign: 'center' }}>
      <h2>O-Ring Size Detection</h2>
      {error && <p style={{ color: 'red' }}>{error}</p>}

      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', maxHeight: '90vh', touchAction: 'none' }}
        onClick={handleCanvasClick}
        onTouchStart={handleCanvasClick}
      />

      <div style={{ marginTop: 10 }}>
        <input
          type="number"
          value={referenceSize}
          onChange={e => setReferenceSize(Number(e.target.value))}
          placeholder="Reference size (mm)"
        />
        {!calibrationComplete ? (
          <>
            <button onClick={startCalibration}>Start Calibration</button>
            <button onClick={completeCalibration}>Finish</button>
          </>
        ) : (
          <button onClick={resetCalibration}>Reset Calibration</button>
        )}
      </div>
    </div>
  );
}

export default App;
