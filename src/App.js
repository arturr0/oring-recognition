import React, { useEffect, useRef, useState } from 'react';

const MODEL_INPUT_SIZE = 640;
const classNames = ['BLOCK', 'INNER', 'OK', 'OUTER', 'SCAR', 'TEAR'];
const NMS_THRESHOLD = 0.5; // Intersection over Union threshold for NMS
const CONFIDENCE_THRESHOLD = 0.4; // Minimum confidence score to consider a detection

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const isProcessingRef = useRef(false);

  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [boxes, setBoxes] = useState([]);

  // Initialize webcam
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
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

  // Initialize worker and load model
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

    const data = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
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

      const x = slice[0];
      const y = slice[1];
      const w = slice[2];
      const h = slice[3];
      const conf = slice[4];
      const classConfs = slice.subarray(5);

      const classId = classConfs.indexOf(Math.max(...classConfs));
      const classConf = classConfs[classId];
      const totalConf = conf * classConf;

      if (totalConf > CONFIDENCE_THRESHOLD) {
        boxes.push({
          x1: x - w / 2,
          y1: y - h / 2,
          x2: x + w / 2,
          y2: y + h / 2,
          label: classNames[classId] || 'unknown',
          confidence: totalConf,
          classId: classId
        });
      }
    }

    return boxes;
  }

  function nonMaxSuppression(boxes) {
    // Sort boxes by confidence in descending order
    const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selectedBoxes = [];

    while (sortedBoxes.length > 0) {
      // Take the box with highest confidence
      const currentBox = sortedBoxes.shift();
      selectedBoxes.push(currentBox);

      // Filter out boxes that have high IoU with the current box and same class
      for (let i = sortedBoxes.length - 1; i >= 0; i--) {
        if (sortedBoxes[i].classId === currentBox.classId) {
          const iou = calculateIoU(currentBox, sortedBoxes[i]);
          if (iou > NMS_THRESHOLD) {
            sortedBoxes.splice(i, 1);
          }
        }
      }
    }

    return selectedBoxes;
  }

  function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  // Drawing + inference loop
  useEffect(() => {
    if (status !== 'ready' || !videoRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    let animationFrameId;

    const run = () => {
      if (!video || video.readyState < 2) {
        animationFrameId = requestAnimationFrame(run);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Draw video frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw detection boxes
      ctx.strokeStyle = 'lime';
      ctx.lineWidth = 2;
      ctx.font = '14px Arial';
      ctx.fillStyle = 'lime';

      boxes.forEach(box => {
        const x = (box.x1 / MODEL_INPUT_SIZE) * canvas.width;
        const y = (box.y1 / MODEL_INPUT_SIZE) * canvas.height;
        const width = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * canvas.width;
        const height = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * canvas.height;

        ctx.strokeRect(x, y, width, height);
        ctx.fillText(`${box.label} (${(box.confidence * 100).toFixed(1)}%)`, x, y - 4);
      });

      // Run inference only if not already processing
      if (!isProcessingRef.current) {
        isProcessingRef.current = true;
        const tensorData = preprocess(video);
        workerRef.current.postMessage({
          type: 'infer',
          tensorData,
          dims: [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        }, [tensorData]);
      }

      animationFrameId = requestAnimationFrame(run);
    };

    animationFrameId = requestAnimationFrame(run);

    return () => cancelAnimationFrame(animationFrameId);
  }, [status, boxes]);

  return (
    <div>
      <video
        ref={videoRef}
        style={{ display: 'none' }}
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', border: '1px solid #aaa' }}
      />
      <div>Status: {status}</div>
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  );
}

export default App;