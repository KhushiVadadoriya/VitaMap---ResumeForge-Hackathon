import { Sparkles, Cpu, Layers, Zap } from "lucide-react";

export function Hero() {
  return (
    <div className="border-b border-slate-200/80 bg-gradient-to-b from-white via-slate-50/40 to-slate-100/30 pt-12 pb-12 sm:pt-16 sm:pb-16">
      <div className="mx-auto max-w-4xl px-4 text-center sm:px-6 lg:px-8">
        <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3.5 py-1 shadow-xs">
          <Sparkles className="h-3.5 w-3.5 text-blue-600" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
            AI Resume Classification
          </span>
        </div>

        <h1 className="mb-4 text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
          Turn a Resume{" "}
          <span className="block text-slate-900">into a Category.</span>
        </h1>

        <p className="mx-auto max-w-2xl text-base text-slate-600 sm:text-lg">
          Analyze resume content using an NLP classification pipeline and identify the most appropriate resume category.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-2.5 text-xs font-medium text-slate-600">
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200/80 bg-white px-3 py-1.5 shadow-xs">
            <Layers className="h-3.5 w-3.5 text-slate-500" />
            NLP Classification
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200/80 bg-white px-3 py-1.5 shadow-xs">
            <Cpu className="h-3.5 w-3.5 text-slate-500" />
            TF-IDF + SVM
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200/80 bg-white px-3 py-1.5 shadow-xs">
            <Zap className="h-3.5 w-3.5 text-slate-500" />
            Real-time inference
          </span>
        </div>
      </div>
    </div>
  );
}

