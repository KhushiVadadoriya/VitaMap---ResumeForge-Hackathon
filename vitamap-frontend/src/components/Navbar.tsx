import { Brain, Radio } from "lucide-react";
import { isMockMode } from "../lib/api";

export function Navbar() {
  const mock = isMockMode();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur-md">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white shadow-xs">
              <Brain className="h-4.5 w-4.5" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-base font-bold tracking-tight text-slate-900">
                VitaMap
              </span>
              <span className="hidden text-xs font-medium text-slate-500 sm:inline-block">
                AI Resume Classification
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {mock ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50/80 px-2.5 py-0.5 text-xs font-medium text-amber-800">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                Mock Mode (Offline)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50/80 px-2.5 py-0.5 text-xs font-medium text-emerald-800">
                <Radio className="h-3 w-3 text-emerald-600 animate-pulse" />
                Live API (localhost:8000)
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}


