import React, { useEffect, useRef, useState } from 'react';

const MODEL_INPUT_SIZE = 640;
const classNames = ['BLOCK', 'INNER', 'OK', 'OUTER', 'SCAR', 'TEAR'];
const NMS_THRESHOLD = 0.5;
const CONFIDENCE_THRESHOLD = 0.1;
const FRAME_SKIP = 2; // Process every 3rd frame on mobile

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const isProcessingRef = useRef(false);
  const frameCountRef = useRef(0);
  const deviceTypeRef = useRef('desktop');

  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [referenceSize, setReferenceSize] = useState(10);
  const [calibrationComplete, setCalibrationComplete] = useState(false);
  const [pixelsPerMM, setPixelsPerMM] = useState(null);
  const [selectedReferenceBox, setSelectedReferenceBox] = useState(null);
  const [orientation, setOrientation] = useState(
    window.screen.orientation?.type || 'landscape-primary'
  );

  // Detect device type and orientation
  useEffect(() => {
    deviceTypeRef.current = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) 
      ? 'mobile' 
      : 'desktop';

    const handleOrientationChange = () => {
      setOrientation(window.screen.orientation?.type || 'landscape-primary');
    };

    window.addEventListener('orientationchange', handleOrientationChange);
    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, []);

  // Initialize webcam with proper constraints for mobile
  useEffect(() => {
    const constraints = {
      video: { 
        facingMode: 'environment',
        width: { ideal: MODEL_INPUT_SIZE },
        height: { ideal: MODEL_INPUT_SIZE },
        frameRate: { ideal: deviceTypeRef.current === 'mobile' ? 15 : 30 }
      }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play().catch(e => {
              setError('Playback error: ' + e.message);
              setStatus('error');
            });
          };
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
    const offscreen = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const ctx = offscreen.getContext('2d', { willReadFrequently: false });
    
    // Maintain aspect ratio while resizing
    const aspectRatio = frame.videoWidth / frame.videoHeight;
    let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
    
    if (aspectRatio > 1) {
      drawWidth = MODEL_INPUT_SIZE;
      drawHeight = MODEL_INPUT_SIZE / aspectRatio;
      offsetY = (MODEL_INPUT_SIZE - drawHeight) / 2;
    } else {
      drawHeight = MODEL_INPUT_SIZE;
      drawWidth = MODEL_INPUT_SIZE * aspectRatio;
      offsetX = (MODEL_INPUT_SIZE - drawWidth) / 2;
    }
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    ctx.drawImage(frame, offsetX, offsetY, drawWidth, drawHeight);
    
    const imgData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
    const data = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    
    // Normalize pixel values to 0-1 and convert to RGB format
    for (let i = 0; i < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; i++) {
      data[i] = imgData[i * 4] / 255.0;         // R
      data[i + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i * 4 + 1] / 255.0; // G
      data[i + 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i * 4 + 2] / 255.0; // B
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
          width: w,
          height: h,
          label: classNames[classId] || 'unknown',
          confidence: totalConf,
          classId: classId
        });
      }
    }

    return boxes;
  }

  function nonMaxSuppression(boxes) {
    const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selectedBoxes = [];

    while (sortedBoxes.length > 0) {
      const currentBox = sortedBoxes.shift();
      selectedBoxes.push(currentBox);

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

  function calculatePhysicalSize(box, ppm) {
    if (!ppm || isNaN(ppm)) return null;
    
    const widthPx = box.width * MODEL_INPUT_SIZE;
    const heightPx = box.height * MODEL_INPUT_SIZE;
    const diameterPx = (widthPx + heightPx) / 2;
    return diameterPx / ppm;
  }

  function handleBoxClick(box) {
    if (calibrationMode) {
      setSelectedReferenceBox(box);
    }
  }

  function startCalibration() {
    if (referenceSize > 0 && !isNaN(referenceSize)) {
      setCalibrationMode(true);
      setError(null);
    } else {
      setError('Please enter a valid reference size first');
    }
  }

  function completeCalibration() {
    if (selectedReferenceBox && referenceSize > 0 && !isNaN(referenceSize)) {
      const widthPx = selectedReferenceBox.width * MODEL_INPUT_SIZE;
      const heightPx = selectedReferenceBox.height * MODEL_INPUT_SIZE;
      const diameterPx = (widthPx + heightPx) / 2;
      const ppm = diameterPx / referenceSize;
      
      setPixelsPerMM(ppm);
      setCalibrationComplete(true);
      setCalibrationMode(false);
      setError(null);
    } else {
      setError('Please select a reference object and ensure valid size');
    }
  }

  function resetCalibration() {
    setCalibrationComplete(false);
    setPixelsPerMM(null);
    setSelectedReferenceBox(null);
    setCalibrationMode(false);
  }

  // Drawing + inference loop with mobile optimizations
  useEffect(() => {
    if (status !== 'ready' || !videoRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: false });

    let animationFrameId;
    let lastInferenceTime = 0;

    const run = () => {
      if (!video || video.readyState < 2) {
        animationFrameId = requestAnimationFrame(run);
        return;
      }

      // Adjust canvas size to match video feed
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Draw video frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw detection boxes
      boxes.forEach(box => {
        const x = (box.x1 / MODEL_INPUT_SIZE) * canvas.width;
        const y = (box.y1 / MODEL_INPUT_SIZE) * canvas.height;
        const width = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * canvas.width;
        const height = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * canvas.height;

        // Highlight selected reference box
        if (calibrationMode && selectedReferenceBox === box) {
          ctx.strokeStyle = 'yellow';
          ctx.lineWidth = 4;
        } else {
          ctx.strokeStyle = box.label === 'OK' ? 'lime' : 'red';
          ctx.lineWidth = 2;
        }

        ctx.strokeRect(x, y, width, height);
        
        // Add label and size information
        let labelText = `${box.label} (${(box.confidence * 100).toFixed(1)}%)`;
        
        if (box.label === 'OK' && pixelsPerMM && !isNaN(pixelsPerMM)) {
          const sizeMM = calculatePhysicalSize(box, pixelsPerMM);
          if (sizeMM) {
            labelText += ` - Ø${sizeMM.toFixed(1)}mm`;
          }
        }

        ctx.fillStyle = 'white';
        ctx.fillRect(x - 2, y - 18, ctx.measureText(labelText).width + 4, 18);
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText(labelText, x, y - 4);
      });

      // Run inference only if not already processing and enough frames have passed
      frameCountRef.current++;
      if (!isProcessingRef.current && frameCountRef.current % (deviceTypeRef.current === 'mobile' ? FRAME_SKIP + 1 : 1) === 0) {
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
  }, [status, boxes, calibrationMode, selectedReferenceBox, pixelsPerMM]);

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ marginBottom: '20px' }}>
        <h1>O-Ring Size Detection</h1>
        
        {!calibrationComplete ? (
          <div style={{ background: '#f0f0f0', padding: '10px', borderRadius: '5px', marginBottom: '10px' }}>
            <h3>Calibration Required</h3>
            <p>Place a reference object of known size in view and enter its diameter:</p>
            
            <div style={{ marginBottom: '10px' }}>
              <label>Reference diameter (mm): </label>
              <input 
                type="number" 
                value={isNaN(referenceSize) ? '' : referenceSize}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  setReferenceSize(isNaN(value) ? 0 : value);
                }} 
                step="0.1"
                min="1"
                style={{ marginLeft: '10px' }}
              />
            </div>
            
            {!calibrationMode ? (
              <button onClick={startCalibration}>Start Calibration</button>
            ) : (
              <div>
                <p>Click on the reference object in the video feed</p>
                {selectedReferenceBox && (
                  <button onClick={completeCalibration}>Complete Calibration</button>
                )}
              </div>
            )}
          </div>
        ) : (
          <div style={{ background: '#e0ffe0', padding: '10px', borderRadius: '5px', marginBottom: '10px' }}>
            <p>Calibration complete: {pixelsPerMM?.toFixed(2) || 'N/A'} pixels/mm</p>
            <button onClick={resetCalibration}>Recalibrate</button>
          </div>
        )}
      </div>

      <video
        ref={videoRef}
        style={{ display: 'none' }}
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', border: '1px solid #aaa', marginBottom: '10px' }}
        onClick={(e) => {
          if (!calibrationMode || !boxes.length) return;
          
          const rect = canvasRef.current.getBoundingClientRect();
          const scaleX = canvasRef.current.width / rect.width;
          const scaleY = canvasRef.current.height / rect.height;
          
          const x = (e.clientX - rect.left) * scaleX;
          const y = (e.clientY - rect.top) * scaleY;
          
          // Find clicked box with tolerance
          const clickedBox = boxes.find(box => {
            const boxX = (box.x1 / MODEL_INPUT_SIZE) * canvasRef.current.width;
            const boxY = (box.y1 / MODEL_INPUT_SIZE) * canvasRef.current.height;
            const boxWidth = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * canvasRef.current.width;
            const boxHeight = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * canvasRef.current.height;
            
            return x >= boxX - 10 && 
                   x <= boxX + boxWidth + 10 && 
                   y >= boxY - 10 && 
                   y <= boxY + boxHeight + 10;
          });
          
          if (clickedBox) {
            handleBoxClick(clickedBox);
          }
        }}
      />
      
      <div style={{ marginTop: '10px' }}>
        <div>Status: {status}</div>
        {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
      </div>
    </div>
  );
}

export default App;
