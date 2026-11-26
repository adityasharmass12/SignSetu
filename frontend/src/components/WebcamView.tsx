import { useCallback, useEffect, useRef, useState } from 'react'
import { HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { getHandLandmarker } from '../lib/handLandmarker'
import { normalizeLandmarks } from '../lib/normalize'

interface WebcamViewProps {
  onFeatures?: (features: Float32Array, handsDetected: number) => void
}

type CameraStatus = 'loading' | 'ready' | 'error'

function getCameraErrorMessage(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      return 'Camera permission was denied. Allow camera access in your browser settings and retry.'
    }
    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      return 'No camera was found on this device. Connect a webcam and retry.'
    }
    if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      return 'Camera is currently in use by another application. Close it and retry.'
    }
  }
  if (err instanceof Error) {
    return err.message
  }
  return 'Unable to access the camera.'
}

function drawHandOverlay(
  ctx: CanvasRenderingContext2D,
  hands: NormalizedLandmark[][],
  width: number,
  height: number
) {
  ctx.clearRect(0, 0, width, height)

  for (const landmarks of hands) {
    ctx.strokeStyle = 'rgba(167, 139, 250, 0.85)'
    ctx.lineWidth = 2.5
    ctx.beginPath()

    for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
      const a = landmarks[start]
      const b = landmarks[end]
      if (!a || !b) continue
      ctx.moveTo(a.x * width, a.y * height)
      ctx.lineTo(b.x * width, b.y * height)
    }
    ctx.stroke()

    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i]
      const x = lm.x * width
      const y = lm.y * height
      ctx.beginPath()
      ctx.arc(x, y, i === 0 ? 5 : 3.5, 0, Math.PI * 2)
      ctx.fillStyle = i === 0 ? '#c4b5fd' : '#7c3aed'
      ctx.fill()
      ctx.strokeStyle = '#f5f3ff'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }
}

export function WebcamView({ onFeatures }: WebcamViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const onFeaturesRef = useRef(onFeatures)

  useEffect(() => {
    onFeaturesRef.current = onFeatures
  }, [onFeatures])

  const [status, setStatus] = useState<CameraStatus>('loading')
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [fps, setFps] = useState<number>(0)
  const [handsDetected, setHandsDetected] = useState<number>(0)
  const [retryCount, setRetryCount] = useState<number>(0)

  const handleRetry = useCallback(() => {
    setRetryCount((c) => c + 1)
  }, [])

  useEffect(() => {
    let active = true
    let animationFrameId = 0
    let stream: MediaStream | null = null
    let lastVideoTime = -1
    let frameCounter = 0
    let fpsWindowStart = performance.now()

    async function start() {
      setStatus('loading')
      setErrorMessage('')
      setHandsDetected(0)
      setFps(0)

      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('error')
        setErrorMessage('Camera API is not supported in this browser or context.')
        return
      }

      try {
        const [landmarker, mediaStream] = await Promise.all([
          getHandLandmarker(),
          navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: 'user',
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          }),
        ])

        if (!active) {
          mediaStream.getTracks().forEach((t) => t.stop())
          return
        }

        stream = mediaStream
        const video = videoRef.current
        if (!video) return

        video.srcObject = mediaStream
        await video.play()

        if (!active) return
        setStatus('ready')

        const renderFrame = () => {
          if (!active) return

          const canvas = canvasRef.current
          if (video.readyState >= 2 && canvas) {
            if (
              canvas.width !== video.videoWidth ||
              canvas.height !== video.videoHeight
            ) {
              canvas.width = video.videoWidth
              canvas.height = video.videoHeight
            }

            const now = performance.now()
            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime
              const result = landmarker.detectForVideo(video, now)
              const ctx = canvas.getContext('2d')
              if (ctx) {
                drawHandOverlay(ctx, result.landmarks, canvas.width, canvas.height)
              }

              const detected = Math.min(result.landmarks.length, 2)
              setHandsDetected(detected)

              if (onFeaturesRef.current) {
                const features = normalizeLandmarks(
                  result.landmarks,
                  result.handedness
                )
                onFeaturesRef.current(features, detected)
              }

              frameCounter += 1
              const elapsed = now - fpsWindowStart
              if (elapsed >= 500) {
                setFps(Math.round((frameCounter * 1000) / elapsed))
                frameCounter = 0
                fpsWindowStart = now
              }
            }
          }

          animationFrameId = requestAnimationFrame(renderFrame)
        }

        animationFrameId = requestAnimationFrame(renderFrame)
      } catch (err) {
        if (!active) return
        setStatus('error')
        setErrorMessage(getCameraErrorMessage(err))
      }
    }

    start()

    return () => {
      active = false
      cancelAnimationFrame(animationFrameId)
      if (stream) {
        stream.getTracks().forEach((t) => t.stop())
      }
    }
  }, [retryCount])

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
              handsDetected > 0
                ? 'bg-violet-500/15 text-violet-300 border-violet-500/40'
                : 'bg-neutral-900 text-neutral-400 border-neutral-800'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                handsDetected > 0 ? 'bg-violet-400' : 'bg-neutral-600'
              }`}
            />
            Hands detected: {handsDetected}
          </span>

          <span className="px-3 py-1 rounded-full text-xs font-mono bg-neutral-900 text-neutral-300 border border-neutral-800">
            {fps} FPS
          </span>
        </div>

        <span className="text-xs text-neutral-500">
          {status === 'loading' && 'Initializing hand tracker...'}
          {status === 'ready' && 'Tracking active (mirrored)'}
          {status === 'error' && 'Camera unavailable'}
        </span>
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900 shadow-2xl shadow-violet-950/20">
        <video
          ref={videoRef}
          playsInline
          muted
          className="h-full w-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />

        {status === 'loading' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-950/80 backdrop-blur-sm gap-3">
            <div className="h-8 w-8 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
            <p className="text-sm text-neutral-300">
              Loading hand landmarker and camera...
            </p>
          </div>
        )}

        {status === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-950/90 p-6 text-center gap-4">
            <div className="max-w-md space-y-2">
              <p className="text-base font-medium text-neutral-100">
                Camera Error
              </p>
              <p className="text-sm text-neutral-400">{errorMessage}</p>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-violet-600 hover:bg-violet-500 text-white transition-colors cursor-pointer"
            >
              Retry Camera
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
