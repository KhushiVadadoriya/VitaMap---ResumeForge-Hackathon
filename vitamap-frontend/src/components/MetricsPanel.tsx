import { useEffect, useState } from "react";
import { BarChart3, TrendingUp, Target, Scale } from "lucide-react";
import type { MetricsResponse } from "../lib/types";
import { getMetrics, isMockMode } from "../lib/api";

interface MetricsPanelProps {
  metrics?: MetricsResponse | null;
}

export function MetricsPanel({ metrics: initialMetrics }: MetricsPanelProps = {}) {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(initialMetrics || null);
  const [loading, setLoading] = useState(!initialMetrics);
  const [error, setError] = useState<string | null>(null);
  const isMock = isMockMode();

  useEffect(() => {
    if (initialMetrics) {
      setMetrics(initialMetrics);
      setLoading(false);
      return;
    }

    async function fetchMetrics() {
      try {
        const data = await getMetrics();
        setMetrics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metrics");
      } finally {
        setLoading(false);
      }
    }
    fetchMetrics();
  }, [initialMetrics]);

  if (loading) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="rounded-2xl border border-slate-200/90 bg-white p-8 text-center shadow-xs">
          <p className="text-xs font-mono text-slate-500">Loading benchmark metrics...</p>
        </div>
      </div>
    );
  }

  if (error || !metrics || !metrics.models || metrics.models.length === 0) {
    return null;
  }

  const primaryModel = metrics.models[0];

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-sm">
        {/* Section Header */}
        <div className="border-b border-slate-100 p-6 sm:p-8">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Evaluation Metrics
            </span>
          </div>
          <h3 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
            Model Intelligence
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Performance across evaluated classification approaches.
          </p>
        </div>

        <div className="p-6 sm:p-8">
          {/* Top 3 Metric Cards */}
          <div className="mb-8 grid gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Accuracy
                </span>
                <Target className="h-4 w-4 text-blue-600" />
              </div>
              <p className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900">
                {(primaryModel.accuracy * 100).toFixed(1)}%
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {primaryModel.name}
              </p>
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Macro-F1
                </span>
                <TrendingUp className="h-4 w-4 text-emerald-600" />
              </div>
              <p className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900">
                {primaryModel.macro_f1.toFixed(2)}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Unweighted class average
              </p>
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Weighted-F1
                </span>
                <Scale className="h-4 w-4 text-purple-600" />
              </div>
              <p className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900">
                {(primaryModel.weighted_f1 ?? primaryModel.macro_f1).toFixed(2)}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Support-weighted score
              </p>
            </div>
          </div>

          {/* Model Comparison Table */}
          <div className="rounded-xl border border-slate-200/80 bg-white">
            <div className="border-b border-slate-100 px-5 py-3.5">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-600">
                Model Comparison
              </h4>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50/50 text-xs font-semibold text-slate-600">
                    <th className="py-3 px-5">Model Architecture</th>
                    <th className="py-3 px-4 text-right">Accuracy</th>
                    <th className="py-3 px-4 text-right">Macro-F1</th>
                    <th className="py-3 px-5 text-right">Weighted-F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {metrics.models.map((model, index) => {
                    const isTop = index === 0;
                    return (
                      <tr key={index} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-3.5 px-5 font-medium text-slate-900">
                          <div className="flex items-center gap-2">
                            <span>{model.name}</span>
                            {isTop && (
                              <span className="inline-flex rounded-full bg-blue-100 px-2 py-0.5 text-2xs font-semibold text-blue-700">
                                Primary
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3.5 px-4 text-right font-mono text-slate-700">
                          {(model.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-4 text-right font-mono text-slate-700">
                          {model.macro_f1.toFixed(2)}
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono text-slate-700">
                          {model.weighted_f1 !== undefined ? model.weighted_f1.toFixed(2) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {isMock && (
            <p className="mt-4 text-right text-2xs text-slate-400 font-mono">
              * Benchmark values sourced from evaluation test partition (Mock mode active).
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
