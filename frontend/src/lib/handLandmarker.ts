import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

let landmarkerPromise: Promise<HandLandmarker> | null = null

async function createHandLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL)

  try {
    return await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    })
  } catch {
    return await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    })
  }
}

export function getHandLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = createHandLandmarker().catch((err) => {
      landmarkerPromise = null
      throw err
    })
  }
  return landmarkerPromise
}
