import { Grid3x3 } from "lucide-react";

interface ConfusionMatrixProps {
  labels: string[];
  matrix: number[][];
}

export function ConfusionMatrix({ labels, matrix }: ConfusionMatrixProps) {
  if (!labels || !matrix || labels.length === 0 || matrix.length === 0) {
    return null;
  }

  // Find max value for color intensity
  const maxValue = Math.max(...matrix.flat(), 1);

  const getCellClasses = (value: number, isDiagonal: boolean): string => {
    const ratio = value / maxValue;
    if (value === 0) {
      return "bg-slate-50 text-slate-300 font-normal";
    }
    if (ratio > 0.7) {
      return "bg-blue-600 text-white font-bold shadow-2xs";
    }
    if (ratio > 0.4) {
      return "bg-blue-400 text-white font-semibold";
    }
    if (ratio > 0.15) {
      return "bg-blue-200 text-slate-900 font-semibold";
    }
    return isDiagonal ? "bg-blue-100 text-blue-900 font-medium" : "bg-blue-50 text-blue-800 font-medium";
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-sm">
        {/* Header */}
        <div className="border-b border-slate-100 p-6 sm:p-8">
          <div className="flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 text-blue-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Error Analysis
            </span>
          </div>
          <h3 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
            Confusion Matrix
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Actual vs predicted category performance.
          </p>
        </div>

        <div className="p-6 sm:p-8">
          {/* Axis Header */}
          <div className="mb-4 text-center">
            <span className="inline-flex items-center rounded-md bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-slate-600">
              Predicted Category →
            </span>
          </div>

          <div className="overflow-x-auto pb-4">
            <div className="inline-block min-w-full align-middle">
              <table className="min-w-full border-separate border-spacing-1.5 text-center">
                <thead>
                  <tr>
                    <th className="w-36 p-2 text-right">
                      <span className="text-2xs font-semibold uppercase tracking-wider text-slate-400">
                        Actual Class ↓
                      </span>
                    </th>
                    {labels.map((label, index) => (
                      <th
                        key={index}
                        className="min-w-[90px] max-w-[120px] p-2 text-center text-xs font-semibold text-slate-700"
                        title={label}
                      >
                        <div className="truncate">{label}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      <td className="p-2 text-right text-xs font-semibold text-slate-700">
                        <div className="truncate" title={labels[rowIndex]}>
                          {labels[rowIndex]}
                        </div>
                      </td>
                      {row.map((value, colIndex) => {
                        const isDiagonal = rowIndex === colIndex;
                        return (
                          <td
                            key={colIndex}
                            className={`rounded-lg p-3 text-center text-sm font-mono transition-colors duration-150 ${getCellClasses(
                              value,
                              isDiagonal
                            )} ${isDiagonal ? "ring-2 ring-blue-500/20" : ""}`}
                            title={`Actual: ${labels[rowIndex]}, Predicted: ${labels[colIndex]} (${value})`}
                          >
                            {value}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Footnote */}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4 text-xs text-slate-500">
            <div className="flex items-center gap-2">
              <span className="inline-block h-3 w-3 rounded bg-blue-600" />
              <span>Diagonal highlights true positive counts (correct predictions)</span>
            </div>
            <span className="font-mono text-2xs text-slate-400">
              Support classes: {labels.length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
