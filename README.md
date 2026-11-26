# SignSetu

Real-time Indian Sign Language (ISL) recognition web app that converts webcam hand signs into text in the browser.

Repository: https://github.com/adityasharmass12/SignSetu

## Structure

- `frontend/` - React + TypeScript + Vite web app with MediaPipe HandLandmarker and browser inference
- `ml/` - Python environment for dataset preprocessing, training, and model export
- `backend/` - Optional Django REST Framework API (added later)

## Run Instructions

### Frontend

```powershell
cd frontend
npm run dev
```

### ML Environment

```powershell
ml\venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```
