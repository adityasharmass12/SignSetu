import { WebcamView } from './components/WebcamView'

export function App() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col">
      <header className="border-b border-neutral-800/80 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="h-2.5 w-2.5 rounded-full bg-violet-500" />
          <span className="text-lg font-semibold tracking-tight">SignSetu</span>
          <span className="text-xs px-2 py-0.5 rounded bg-violet-500/15 text-violet-300 border border-violet-500/30">
            ISL
          </span>
        </div>
        <span className="text-xs text-neutral-500">In-browser hand tracking</span>
      </header>

      <main className="flex-1 flex items-center justify-center p-6">
        <WebcamView />
      </main>
    </div>
  )
}

export default App
