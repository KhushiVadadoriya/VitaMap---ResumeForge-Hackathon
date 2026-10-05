import { CheckCircle2, Tag, RefreshCw, Cpu, Award } from "lucide-react";
import type { PredictResponse } from "../lib/types";

interface PredictionResultProps {
  prediction: PredictResponse;
  onAnalyzeAnother: () => void;
}

export function PredictionResult({
  prediction,
  onAnalyzeAnother,
}: PredictionResultProps) {
  // Step 6: Truthful Score Display
  // If backend provides a calibrated probability in [0, 1] or XX%, display "Confidence: XX%"
  // Otherwise display "Decision Score: VALUE" without fabricating confidence
  const getScoreDisplay = (score: string | number): { label: string; value: string } => {
    if (typeof score === "number") {
      if (score >= 0 && score <= 1) {
        return {
          label: "Confidence",
          value: `${Math.round(score * 100)}%`,
        };
      }
      return {
        label: "Decision Score",
        value: score.toFixed(2),
      };
    }

    if (typeof score === "string") {
      const trimmed = score.trim();
      if (trimmed.endsWith("%")) {
        return {
          label: "Confidence",
          value: trimmed,
        };
      }
      const num = parseFloat(trimmed);
      if (!isNaN(num)) {
        if (num >= 0 && num <= 1) {
          return {
            label: "Confidence",
            value: `${Math.round(num * 100)}%`,
          };
        }
        return {
          label: "Decision Score",
          value: num.toFixed(2),
        };
      }
      return {
        label: "Decision Score",
        value: trimmed,
      };
    }

    return {
      label: "Decision Score",
      value: String(score),
    };
  };

  const scoreInfo = getScoreDisplay(prediction.confidence);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      {/* High-Impact Result Card */}
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-sm transition-all">
        {/* Card Header Status */}
        <div className="border-b border-emerald-100 bg-emerald-50/70 px-6 py-4 sm:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-emerald-800">
              <CheckCircle2 className="h-5 w-5 text-emerald-600" />
              <span className="text-xs font-bold uppercase tracking-wider">
                Analysis Complete
              </span>
            </div>
            <span className="text-xs font-mono text-emerald-700">
              Inference 200 OK
            </span>
          </div>
        </div>

        <div className="p-6 sm:p-10">
          {/* Main Predicted Category */}
          <div className="mb-8 text-center sm:mb-10">
            <p className="mb-2 text-xs font-bold uppercase tracking-widest text-slate-500">
              Predicted Category
            </p>
            <h2 className="text-3xl font-extrabold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              {prediction.category}
            </h2>
          </div>

          {/* Metadata Grid */}
          <div className="mb-8 grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-200/80 bg-slate-50/60 p-5">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-slate-500" />
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Model
                </p>
              </div>
              <p className="mt-2 text-base font-semibold text-slate-900 font-mono">
                {prediction.model}
              </p>
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-slate-50/60 p-5">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-blue-600" />
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  {scoreInfo.label}
                </p>
              </div>
              <p className="mt-2 text-2xl font-extrabold text-blue-600">
                {scoreInfo.value}
              </p>
            </div>
          </div>

          {/* Top Terms (If present) */}
          {prediction.top_terms && prediction.top_terms.length > 0 && (
            <div className="mb-8 rounded-xl border border-slate-200/80 bg-slate-50/30 p-5">
              <div className="mb-3 flex items-center gap-2">
                <Tag className="h-4 w-4 text-slate-500" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">
                  Key Terms Extracted
                </h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {prediction.top_terms.map((term, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-2xs"
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                    {term}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Action Footer */}
          <div className="border-t border-slate-100 pt-6 text-center">
            <button
              type="button"
              onClick={onAnalyzeAnother}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white shadow-xs transition-all hover:bg-blue-700 active:scale-98"
            >
              <RefreshCw className="h-4 w-4" />
              Analyze Another Resume
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
