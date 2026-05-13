import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import './CameraScreen.css'

const MODEL_PATH = '/model/model.json?v=' + Date.now()
const THRESHOLD = 0.5
const TOTAL_FRAMES = 5

export default function CameraScreen({ onResult, onBack }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const modelRef = useRef(null)

  const [mode, setMode] = useState('camera') // 'camera' | 'upload'
  const [hasPermission, setHasPermission] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [modelError, setModelError] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [frameCount, setFrameCount] = useState(0)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [loadingModel, setLoadingModel] = useState(true)

  // Upload mode state
  const [uploadedImage, setUploadedImage] = useState(null) // data URL
  const [uploadedCanvas, setUploadedCanvas] = useState(null) // offscreen canvas
  const [analyzing, setAnalyzing] = useState(false)

  // Load model on mount
  useEffect(() => {
    loadModel()
  }, [])

  // Start/stop camera based on mode
  useEffect(() => {
    if (mode === 'camera') {
      startCamera()
    } else {
      stopCamera()
      setUploadedImage(null)
      setUploadedCanvas(null)
    }
    return () => { }
  }, [mode])

  // Cleanup camera on unmount
  useEffect(() => {
    return () => stopCamera()
  }, [])

  const loadModel = async () => {
    try {
      setLoadingModel(true)
      setModelError(null)
      // Use WASM backend — the .wasm files are served from /public
      setWasmPaths('/')
      await tf.setBackend('wasm')
      await tf.ready()
      console.log('Backend:', tf.getBackend())
      const model = await tf.loadGraphModel(MODEL_PATH)
      modelRef.current = model
      setModelLoaded(true)
      console.log('✓ Model loaded successfully with WASM backend')
    } catch (err) {
      console.error('WASM backend error:', err)
      try {
        await tf.setBackend('cpu')
        await tf.ready()
        const model = await tf.loadGraphModel(MODEL_PATH)
        modelRef.current = model
        setModelLoaded(true)
        console.log('✓ Model loaded with CPU fallback')
      } catch (err2) {
        console.error('CPU fallback error:', err2)
        setModelError(`Failed to load AI model: ${err2.message || 'Unknown error'}`)
      }
    } finally {
      setLoadingModel(false)
    }
  }

  const startCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setHasPermission(true)
      }
    } catch (err) {
      setCameraError('Camera access denied. Please allow camera permission and refresh.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setHasPermission(false)
  }

  // ── Image quality checks ───────────────────────────────────────
  const isBlurry = (imageData) => {
    const data = imageData.data
    let sum = 0, sumSq = 0
    const count = data.length / 4
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      sum += gray
      sumSq += gray * gray
    }
    const mean = sum / count
    const variance = (sumSq / count) - (mean * mean)
    return variance < 500
  }

  const isDark = (imageData) => {
    const data = imageData.data
    let total = 0
    for (let i = 0; i < data.length; i += 4) {
      total += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    return (total / (data.length / 4)) < 40
  }

  // ── Inference ─────────────────────────────────────────────────
  const runInference = async (originalCanvas) => {
    if (!modelRef.current) return null
    const model = modelRef.current

    // Apply a pre-inference radial vignette mask — forces the model
    // to focus on the centre and ignore peripheral background noise.
    const maskedCanvas = document.createElement('canvas')
    maskedCanvas.width = 224; maskedCanvas.height = 224
    const mCtx = maskedCanvas.getContext('2d')
    mCtx.drawImage(originalCanvas, 0, 0)
    mCtx.globalCompositeOperation = 'source-over'
    const grad = mCtx.createRadialGradient(112, 112, 60, 112, 112, 112)
    grad.addColorStop(0, 'rgba(0,0,0,0)')
    grad.addColorStop(1, 'rgba(0,0,0,1)')
    mCtx.fillStyle = grad
    mCtx.fillRect(0, 0, 224, 224)

    // Build the input tensor (kept alive outside tidy so grad can use it)
    const inputTensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(maskedCanvas)
      t = tf.image.resizeBilinear(t, [224, 224])
      return t.toFloat().div(127.5).sub(1.0).expandDims(0)
    })

    // Forward pass — get confidence score
    const predTensor = model.predict(inputTensor)
    const confidence = predTensor.dataSync()[0]
    predTensor.dispose()

    // ── Gradient Saliency Map ──────────────────────────────────────
    // Compute d(output)/d(input) — pixels with large gradient magnitude
    // had the most influence on the prediction, so they form the real
    // lesion mask rather than the synthetic circle fallback.
    let heatmapData = null
    try {
      const gradFn = tf.grad((x) => model.predict(x).squeeze())
      const saliency = gradFn(inputTensor)           // shape [1, 224, 224, 3]

      heatmapData = tf.tidy(() => {
        // Absolute gradient, max across RGB channels → [224, 224]
        let hm = saliency.abs().squeeze().max(-1)

        // Apply a gentle centre-bias to suppress boundary noise
        const [h, w] = [224, 224]
        const biasData = new Float32Array(h * w)
        for (let r = 0; r < h; r++) {
          for (let c = 0; c < w; c++) {
            const dy = (r / (h - 1)) - 0.5
            const dx = (c / (w - 1)) - 0.5
            biasData[r * w + c] = Math.exp(-(dx * dx + dy * dy) * 3.0)
          }
        }
        hm = hm.mul(tf.tensor2d(biasData, [h, w]))

        // Normalise [0, 1]
        const minV = hm.min()
        const maxV = hm.max()
        hm = hm.sub(minV).div(maxV.sub(minV).add(1e-7))
        return hm.dataSync()           // Float32Array
      })

      saliency.dispose()
      console.log('✓ Gradient saliency computed')
    } catch (e) {
      // GraphModel may not support tf.grad in all TF.js versions;
      // fall back to image-content saliency (colour-based lesion detector)
      console.warn('Gradient saliency failed, using colour saliency:', e.message)
      heatmapData = computeColorSaliency(originalCanvas)
    }

    inputTensor.dispose()
    return { confidence, heatmapData }
  }

  // ── Camera: capture one frame from video ───────────────────────
  const captureFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    const ctx = canvas.getContext('2d')
    canvas.width = 224
    canvas.height = 224

    const size = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - size) / 2
    const sy = (video.videoHeight - size) / 2
    ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224)

    const imageData = ctx.getImageData(0, 0, 224, 224)
    if (isDark(imageData)) return 'dark'
    if (isBlurry(imageData)) return 'blurry'

    try {
      const inferenceResult = await runInference(canvas)
      if (inferenceResult === null) return null
      return {
        label: inferenceResult.confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE',
        confidence: inferenceResult.confidence,
        heatmapData: inferenceResult.heatmapData
      }
    } catch (e) {
      console.error("Inference Error:", e)
      return null;
    }
  }

  // ── Colour-Content Saliency (real lesion detector) ────────────────
  // Analyses each pixel for the visual signatures of ringworm skin lesions:
  //   • High colour saturation  (inflamed skin is vivid vs dull fur)
  //   • Redness / pinkness      (erythema — hallmark of ringworm)
  //   • Mid-brightness only     (suppress shadows and specular highlights)
  // A gentle centre-bias is then applied because users centre the lesion.
  // This produces an irregularly-shaped heatmap that matches the real lesion
  // instead of the previous perfect synthetic circle.
  const computeColorSaliency = (canvas) => {
    const ctx = canvas.getContext('2d')
    const { data } = ctx.getImageData(0, 0, 224, 224)
    const W = 224, H = 224
    const raw = new Float32Array(W * H)

    for (let i = 0; i < W * H; i++) {
      const r = data[i * 4]
      const g = data[i * 4 + 1]
      const b = data[i * 4 + 2]

      const maxC = Math.max(r, g, b)
      const minC = Math.min(r, g, b)

      // Saturation [0–1] — how vivid the pixel is
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0

      // Redness [0–1] — red dominance over green+blue average
      const redness = maxC > 10 ? Math.max(0, (r * 2 - g - b) / (maxC * 2)) : 0

      // Brightness [0–1]
      const bright = (r + g + b) / 765

      // Suppress very dark (shadow) and very bright (specular highlight) pixels
      const brightMask = Math.min(1, bright * 5) * Math.min(1, (1 - bright) * 5)

      raw[i] = (sat * 0.55 + redness * 0.45) * brightMask
    }

    // Gentle centre-bias: lesion is expected near the image centre
    for (let row = 0; row < H; row++) {
      for (let col = 0; col < W; col++) {
        const dy = (row / (H - 1)) - 0.5
        const dx = (col / (W - 1)) - 0.5
        raw[row * W + col] *= Math.exp(-(dx * dx + dy * dy) * 2.5)
      }
    }

    // Normalise [0, 1]
    let maxVal = 0
    for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i]
    if (maxVal > 0) for (let i = 0; i < raw.length; i++) raw[i] /= maxVal

    return raw
  }

  const drawHeatmap = (sourceCanvas, rawHeatmap) => {
    const width = 224;
    const height = 224;

    // Apply a Gaussian blur to the heatmap for smooth, MRI-like professional transitions
    const radius = 15;
    const sigma = radius / 3;
    const kernelSize = radius * 2 + 1;
    const kernel = new Float32Array(kernelSize);
    let sum = 0;
    for (let i = 0; i < kernelSize; i++) {
      const x = i - radius;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    for (let i = 0; i < kernelSize; i++) kernel[i] /= sum;

    const temp = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const px = Math.min(Math.max(x + k, 0), width - 1);
          val += rawHeatmap[y * width + px] * kernel[k + radius];
        }
        temp[y * width + x] = val;
      }
    }

    const heatmap = new Float32Array(width * height);
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const py = Math.min(Math.max(y + k, 0), height - 1);
          val += temp[py * width + x] * kernel[k + radius];
        }
        heatmap[y * width + x] = val;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    // Normalize heatmap [0, 1] after blurring
    const range = maxVal - minVal || 1;
    for (let i = 0; i < heatmap.length; i++) {
      heatmap[i] = (heatmap[i] - minVal) / range;
    }

    const out = document.createElement('canvas')
    out.width = width; out.height = height
    const ctx = out.getContext('2d')
    ctx.drawImage(sourceCanvas, 0, 0)
    const imageData = ctx.getImageData(0, 0, width, height)
    const data = imageData.data

    // Jet colormap equivalent to cv2.COLORMAP_JET
    const getJetColor = (v) => {
      let r = 0, g = 0, b = 0;
      if (v < 0.125) {
        b = 128 + (v / 0.125) * 127;
      } else if (v < 0.375) {
        b = 255;
        g = ((v - 0.125) / 0.25) * 255;
      } else if (v < 0.625) {
        r = ((v - 0.375) / 0.25) * 255;
        g = 255;
        b = 255 - ((v - 0.375) / 0.25) * 255;
      } else if (v < 0.875) {
        r = 255;
        g = 255 - ((v - 0.625) / 0.25) * 255;
      } else {
        r = 255 - ((v - 0.875) / 0.125) * 127;
      }
      return [Math.round(r), Math.round(g), Math.round(b)];
    }

    for (let i = 0; i < heatmap.length; i++) {
      let val = heatmap[i];
      const p = i * 4;
      const [r, g, b] = getJetColor(val);

      // Convert original pixel to grayscale so the heatmap colors pop exactly like an MRI
      const origR = data[p];
      const origG = data[p + 1];
      const origB = data[p + 2];
      const gray = 0.299 * origR + 0.587 * origG + 0.114 * origB;

      // Adjust alpha blending: 'hot' areas (red/orange) are more opaque (0.9), 
      // while the background retains a prominent blue tint (0.5) over the grayscale image.
      const alpha = 0.5 + (val * 0.4);
      const invAlpha = 1 - alpha;

      data[p] = Math.round(gray * invAlpha + r * alpha);
      data[p + 1] = Math.round(gray * invAlpha + g * alpha);
      data[p + 2] = Math.round(gray * invAlpha + b * alpha);
    }
    ctx.putImageData(imageData, 0, 0)
    return out.toDataURL('image/jpeg', 0.9)
  }

  // ── Lesion Shape Classifier (Fur-nsics Unique Feature) ──────────
  // Computes circularity ratio of the Grad-CAM activation region
  // Circularity = (4 * PI * Area) / (Perimeter^2)
  // Classic Ring Pattern >= 0.75 | Partial Ring 0.40-0.74 | Atypical < 0.40
  const computeLesionMorphology = (heatmapData) => {
    const width = 224, height = 224
    const HEATMAP_THRESHOLD = 0.5

    // Step 1 — Binary threshold the heatmap
    const binary = new Uint8Array(width * height)
    for (let i = 0; i < heatmapData.length; i++) {
      binary[i] = heatmapData[i] >= HEATMAP_THRESHOLD ? 1 : 0
    }

    // Step 2 — Compute Area (count of active pixels)
    let area = 0
    for (let i = 0; i < binary.length; i++) area += binary[i]

    if (area === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 3 — Compute Perimeter (count boundary pixels)
    let perimeter = 0
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!binary[y * width + x]) continue
        const neighbors = [
          [y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]
        ]
        const isBoundary = neighbors.some(([ny, nx]) => {
          if (ny < 0 || ny >= height || nx < 0 || nx >= width) return true
          return binary[ny * width + nx] === 0
        })
        if (isBoundary) perimeter++
      }
    }

    if (perimeter === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 4 — Circularity = (4 * PI * Area) / (Perimeter^2)
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter)

    // Step 5 — Classify LMS category
    let lms
    if (circularity >= 0.75) lms = 'Classic Ring Pattern'
    else if (circularity >= 0.40) lms = 'Partial Ring Pattern'
    else lms = 'Atypical Pattern'

    return { lms, circularity: Math.round(circularity * 100) / 100 }
  }

  // ── Build result object (shared by both modes) ─────────────────
  const buildResult = (finalLabel, avgConfidence, sourceCanvas, frameCount, totalFrames, actualHeatmap) => {
    // Use actual heatmap from gradient saliency; fall back to colour-content
    // saliency derived from the source image — never a synthetic circle.
    const heatmap = actualHeatmap || computeColorSaliency(sourceCanvas)
    const heatmapImage = finalLabel === 'POSITIVE' ? drawHeatmap(sourceCanvas, heatmap) : null

    // Compute Lesion Morphology Score (Fur-nsics unique feature)
    const morphology = finalLabel === 'POSITIVE' && actualHeatmap
      ? computeLesionMorphology(actualHeatmap)
      : { lms: 'N/A', circularity: 0 }

    let displayLabel
    if (finalLabel === 'POSITIVE' && avgConfidence >= 0.75) displayLabel = 'RINGWORM SUSPECTED'
    else if (finalLabel === 'POSITIVE' && avgConfidence >= 0.5) displayLabel = 'NEEDS ATTENTION'
    else displayLabel = 'NO RINGWORM SUSPECTED'

    return {
      label: finalLabel,
      displayLabel,
      confidence: avgConfidence,
      heatmapImage,
      rawImage: sourceCanvas.toDataURL('image/jpeg', 0.95),
      positiveVotes: frameCount,
      totalFrames,
      lms: morphology.lms,
      circularity: morphology.circularity
    }
  }

  // ── Camera mode: 5-frame screening ────────────────────────────
  const runScreening = async () => {
    if (capturing || !modelLoaded) return
    setCapturing(true)
    setFrameCount(0)

    const results = []
    let blurCount = 0, darkCount = 0
    let lastCanvas = null

    for (let i = 0; i < TOTAL_FRAMES; i++) {
      await new Promise(r => setTimeout(r, 500))
      const result = await captureFrame()

      if (result === 'blurry') {
        blurCount++
        if (blurCount >= 3) { setCapturing(false); setFrameCount(0); alert('Too many blurry frames. Hold camera steady and try again.'); return }
        continue
      }
      if (result === 'dark') {
        darkCount++
        if (darkCount >= 3) { setCapturing(false); setFrameCount(0); alert('Too dark. Find better lighting and try again.'); return }
        continue
      }
      if (result) {
        results.push(result)
        if (!lastCanvas) {
          lastCanvas = document.createElement('canvas')
          lastCanvas.width = 224; lastCanvas.height = 224
          lastCanvas.getContext('2d').drawImage(canvasRef.current, 0, 0)
        }
      }
      setFrameCount(i + 1)
    }

    if (results.length === 0) { setCapturing(false); alert('Could not get a clear frame. Please try again.'); return }

    const positiveVotes = results.filter(r => r.label === 'POSITIVE').length
    const avgConfidence = results.reduce((a, b) => a + b.confidence, 0) / results.length
    const finalLabel = positiveVotes >= 3 ? 'POSITIVE' : 'NEGATIVE'

    setCapturing(false)
    stopCamera()

    // Find the last frame that contributed to the positive result to show its heatmap
    const lastPositiveResult = results.slice().reverse().find(r => r.label === 'POSITIVE') || results[results.length - 1]

    onResult(buildResult(finalLabel, avgConfidence, lastCanvas, positiveVotes, results.length, lastPositiveResult?.heatmapData))
  }

  // ── Upload mode: handle file pick ─────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const dataUrl = ev.target.result
      const img = new Image()
      img.onload = () => {
        const offscreen = document.createElement('canvas')
        offscreen.width = 224; offscreen.height = 224
        const ctx = offscreen.getContext('2d')
        // Centre-crop
        const size = Math.min(img.width, img.height)
        const sx = (img.width - size) / 2
        const sy = (img.height - size) / 2
        ctx.drawImage(img, sx, sy, size, size, 0, 0, 224, 224)
        setUploadedImage(dataUrl)
        setUploadedCanvas(offscreen)
      }
      img.src = dataUrl
    }
    reader.readAsDataURL(file)
    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  const runUploadInference = async () => {
    if (!uploadedCanvas || !modelLoaded || analyzing) return
    setAnalyzing(true)

    const ctx = uploadedCanvas.getContext('2d')
    const imageData = ctx.getImageData(0, 0, 224, 224)

    if (isDark(imageData)) {
      setAnalyzing(false)
      alert('Image is too dark. Please use a brighter photo.')
      return
    }

    try {
      const inferenceResult = await runInference(uploadedCanvas)
      if (inferenceResult === null) {
        setAnalyzing(false)
        alert('Could not analyse image. Please try again.')
        return
      }

      const confidence = inferenceResult.confidence;
      const finalLabel = confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE'
      setAnalyzing(false)
      onResult(buildResult(finalLabel, confidence, uploadedCanvas, finalLabel === 'POSITIVE' ? 1 : 0, 1, inferenceResult.heatmapData))
    } catch (e) {
      console.error("Inference error:", e)
      setAnalyzing(false)
      alert('An error occurred during analysis: ' + e.message)
    }
  }

  // ── Render ────────────────────────────────────────────────────
  return (
    <div className="camera-screen">
      {/* Background layer: Always camera, or uploaded image if selected */}
      <div className="viewfinder-wrap">
        {uploadedImage ? (
          <img src={uploadedImage} alt="Selected" className="viewfinder" />
        ) : cameraError ? (
          <div className="cam-error">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '12px' }}>
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
              <line x1="2" y1="2" x2="22" y2="22"></line>
            </svg>
            <p>{cameraError}</p>
          </div>
        ) : (
          <video ref={videoRef} autoPlay playsInline muted className="viewfinder" />
        )}
      </div>

      {/* UI Layer */}
      <div className="ui-layer">
        {/* Header */}
        <div className="ui-header">
          <div className="ui-header-top">
            <h1 className="ui-title">Fur-nsics Scanner</h1>
            <button className="ui-close-btn" onClick={onBack}>✕</button>
          </div>
          <p className="ui-subtitle">Point your camera at a skin lesion and capture it.</p>
        </div>

        {/* Center Area */}
        <div className="ui-center-area">
          {!uploadedImage && (
            <div className="ui-target-box">
              <div className="corner tl" />
              <div className="corner tr" />
              <div className="corner bl" />
              <div className="corner br" />
            </div>
          )}
          
          {!uploadedImage && (
            <button className="ui-upload-pill" onClick={() => fileInputRef.current?.click()}>
              Upload From Gallery 
              <svg className="upload-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </button>
          )}

          {modelError && (
            <div className="model-error-banner">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px' }}>
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
              </svg>
              {modelError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="ui-footer">
          {uploadedImage ? (
            <div className="ui-actions">
              <button className="ui-action-btn secondary" onClick={() => fileInputRef.current?.click()}>
                Change Photo
              </button>
              <button className="ui-action-btn primary" onClick={runUploadInference} disabled={analyzing || loadingModel}>
                {analyzing ? 'Analyzing...' : 'Analyze Photo'}
              </button>
            </div>
          ) : (
            <div className="shutter-btn-wrap">
              <button 
                className={`ui-shutter-btn ${capturing ? 'capturing' : ''}`}
                onClick={runScreening}
                disabled={!hasPermission || capturing || loadingModel}
              >
                <div className="ui-shutter-inner">
                  {capturing && <span className="shutter-progress">{frameCount}/5</span>}
                </div>
              </button>
            </div>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
    </div>
  )
}
