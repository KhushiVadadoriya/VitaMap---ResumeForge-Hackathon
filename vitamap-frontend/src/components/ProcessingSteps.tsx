import { Check, Loader2, FileText, Cpu, Layers, Brain, CheckCircle2 } from "lucide-react";
import type { ProcessingStage } from "../lib/types";

interface ProcessingStepsProps {
  currentStage: ProcessingStage | null;
}

const STAGES: {
  id: ProcessingStage;
  label: string;
  description: string;
  icon: typeof FileText;
}[] = [
  {
    id: "resume_received",
    label: "Resume received",
    description: "Ingesting raw document and validating text format",
    icon: FileText,
  },
  {
    id: "text_preprocessing",
    label: "Text preprocessing",
    description: "Tokenizing words, stripping stopwords, and normalizing vocabulary",
    icon: Cpu,
  },
  {
    id: "feature_extraction",
    label: "Feature extraction",
    description: "Generating TF-IDF vector embeddings and term frequencies",
    icon: Layers,
  },
  {
    id: "model_inference",
    label: "Model inference",
    description: "Passing sparse feature vectors into linear classifier",
    icon: Brain,
  },
  {
    id: "category_identified",
    label: "Category identified",
    description: "Determining target resume class and confidence distribution",
    icon: CheckCircle2,
  },
];

export function ProcessingSteps({ currentStage }: ProcessingStepsProps) {
  const currentIndex = STAGES.findIndex((s) => s.id === currentStage);
  const activeStep = currentIndex >= 0 ? currentIndex : 0;
  const progressPercent =
    currentIndex >= 0 ? Math.round(((currentIndex + 1) / STAGES.length) * 100) : 10;

  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-sm">
        {/* Progress Bar */}
        <div className="h-1.5 w-full bg-slate-100">
          <div
            className="h-full bg-blue-600 transition-all duration-300 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        <div className="p-6 sm:p-8">
          <div className="mb-6 flex items-center justify-between border-b border-slate-100 pb-5">
            <div>
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-blue-600">
                <span className="h-2 w-2 rounded-full bg-blue-600 animate-pulse" />
                Pipeline Active
              </span>
              <h3 className="mt-1 text-lg font-bold tracking-tight text-slate-900">
                Processing Resume
              </h3>
            </div>
            <div className="text-right">
              <span className="font-mono text-xs font-semibold text-slate-500">
                Step {Math.min(activeStep + 1, 5)} / 5
              </span>
            </div>
          </div>

          <div className="relative space-y-5">
            {STAGES.map((stage, index) => {
              const isComplete = index < currentIndex;
              const isCurrent = index === currentIndex;
              const isLast = index === STAGES.length - 1;
              const Icon = stage.icon;

              return (
                <div key={stage.id} className="relative flex items-start gap-4">
                  {/* Vertical Connector Line */}
                  {!isLast && (
                    <div
                      className={`absolute top-9 left-4.5 -ml-px h-full w-0.5 transition-colors duration-300 ${
                        isComplete ? "bg-emerald-500" : "bg-slate-200"
                      }`}
                      style={{ height: "calc(100% - 10px)" }}
                    />
                  )}

                  {/* Stage Icon Node */}
                  <div className="relative z-10 flex-shrink-0">
                    {isComplete ? (
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-600 text-white shadow-xs">
                        <Check className="h-4.5 w-4.5" />
                      </div>
                    ) : isCurrent ? (
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600 text-white shadow-xs ring-4 ring-blue-100">
                        <Loader2 className="h-4.5 w-4.5 animate-spin" />
                      </div>
                    ) : (
                      <div className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-slate-50 text-slate-400">
                        <Icon className="h-4 w-4" />
                      </div>
                    )}
                  </div>

                  {/* Stage Details */}
                  <div className="flex-1 pt-1">
                    <div className="flex items-center justify-between">
                      <p
                        className={`text-sm font-semibold transition-colors ${
                          isComplete
                            ? "text-slate-900"
                            : isCurrent
                            ? "text-blue-600"
                            : "text-slate-400"
                        }`}
                      >
                        {stage.label}
                      </p>
                      <span
                        className={`text-xs font-mono font-medium ${
                          isComplete
                            ? "text-emerald-600"
                            : isCurrent
                            ? "text-blue-600 animate-pulse"
                            : "text-slate-400"
                        }`}
                      >
                        {isComplete ? "Done" : isCurrent ? "Running..." : "Pending"}
                      </span>
                    </div>
                    <p
                      className={`mt-0.5 text-xs transition-colors ${
                        isCurrent
                          ? "text-slate-600"
                          : isComplete
                          ? "text-slate-500"
                          : "text-slate-400/80"
                      }`}
                    >
                      {stage.description}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

