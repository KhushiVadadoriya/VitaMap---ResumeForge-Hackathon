import { useState, useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Navbar } from "./components/Navbar";
import { Hero } from "./components/Hero";
import { ResumeInput } from "./components/ResumeInput";
import { ProcessingSteps } from "./components/ProcessingSteps";
import { PredictionResult } from "./components/PredictionResult";
import { MetricsPanel } from "./components/MetricsPanel";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import type { AppState, ProcessingStage, PredictResponse, MetricsResponse } from "./lib/types";
import { predictResume, getMetrics } from "./lib/api";

const PROCESSING_STAGES: ProcessingStage[] = [
  "resume_received",
  "text_preprocessing",
  "feature_extraction",
  "model_inference",
  "category_identified",
];

function App() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [currentStage, setCurrentStage] = useState<ProcessingStage | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  // Load metrics on mount
  useEffect(() => {
    async function loadMetrics() {
      try {
        const data = await getMetrics();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to load metrics:", err);
      }
    }
    loadMetrics();
  }, []);

  const handleAnalyze = async (resumeText: string) => {
    setAppState("processing");
    setError(null);
    setPrediction(null);

    try {
      // Animate through processing stages
      for (let i = 0; i < PROCESSING_STAGES.length; i++) {
        setCurrentStage(PROCESSING_STAGES[i]);
        await new Promise((resolve) => setTimeout(resolve, 400));
      }

      const result = await predictResume(resumeText);
      setPrediction(result);
      setAppState("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze resume");
      setAppState("error");
    } finally {
      setCurrentStage(null);
    }
  };

  const handleAnalyzeAnother = () => {
    setAppState("idle");
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="flex min-h-screen flex-col bg-slate-50/50 text-slate-900 antialiased">
      <Navbar />

      <main className="flex-1 pb-16">
        {appState === "idle" && (
          <>
            <Hero />
            <ResumeInput onAnalyze={handleAnalyze} disabled={false} />
          </>
        )}

        {appState === "processing" && (
          <>
            <Hero />
            <ProcessingSteps currentStage={currentStage} />
          </>
        )}

        {appState === "success" && prediction && (
          <>
            <Hero />
            <PredictionResult
              prediction={prediction}
              onAnalyzeAnother={handleAnalyzeAnother}
            />
            <MetricsPanel metrics={metrics} />
            {metrics && (
              <ConfusionMatrix
                labels={metrics.labels}
                matrix={metrics.confusion_matrix}
              />
            )}
          </>
        )}

        {appState === "error" && (
          <>
            <Hero />
            <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
              <div className="overflow-hidden rounded-2xl border border-red-200 bg-white p-8 text-center shadow-xs">
                <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-red-100 text-red-600">
                  <AlertCircle className="h-6 w-6" />
                </div>
                <h3 className="text-base font-bold text-slate-900">Analysis Failed</h3>
                <p className="mx-auto mt-2 max-w-md text-xs leading-relaxed text-slate-600">
                  {error || "Unable to analyze the resume. Please ensure the backend server is running, or verify input."}
                </p>
                <div className="mt-6">
                  <button
                    type="button"
                    onClick={handleAnalyzeAnother}
                    className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-xs font-semibold text-white shadow-xs transition-all hover:bg-blue-700 active:scale-98"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Try Again
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      {/* Small Technical Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-6 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <p className="font-semibold text-slate-700">
            VitaMap • AI Resume Classification Engine
          </p>
          <p className="mt-1 text-slate-400">
            Hackathon Demonstration Platform • NLP Inference Pipeline
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

