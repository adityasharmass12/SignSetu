import type { Category, NormalizedLandmark } from '@mediapipe/tasks-vision'

const LANDMARKS_PER_HAND = 21
const COORDS_PER_LANDMARK = 3
export const HAND_FEATURE_SIZE = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
export const TOTAL_FEATURE_SIZE = HAND_FEATURE_SIZE * 2

function writeNormalizedHand(
  hand: NormalizedLandmark[],
  out: Float32Array,
  offset: number
) {
  if (hand.length < LANDMARKS_PER_HAND) return

  const wrist = hand[0]
  let maxDist = 0

  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const dx = hand[i].x - wrist.x
    const dy = hand[i].y - wrist.y
    const dz = hand[i].z - wrist.z
    const dist = Math.hypot(dx, dy, dz)
    if (dist > maxDist) {
      maxDist = dist
    }
  }

  const scale = maxDist > 1e-6 ? maxDist : 1

  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const base = offset + i * COORDS_PER_LANDMARK
    out[base] = (hand[i].x - wrist.x) / scale
    out[base + 1] = (hand[i].y - wrist.y) / scale
    out[base + 2] = (hand[i].z - wrist.z) / scale
  }
}

export function normalizeLandmarks(
  landmarks: NormalizedLandmark[][],
  handedness: Category[][]
): Float32Array {
  const features = new Float32Array(TOTAL_FEATURE_SIZE)
  const filled = [false, false]
  const count = Math.min(landmarks.length, 2)

  for (let i = 0; i < count; i++) {
    const label = handedness[i]?.[0]?.categoryName
    let slot = label === 'Right' ? 1 : 0

    if (filled[slot]) {
      slot = slot === 0 ? 1 : 0
    }

    filled[slot] = true
    writeNormalizedHand(landmarks[i], features, slot * HAND_FEATURE_SIZE)
  }

  return features
}
