import { useState, useRef, type ChangeEvent, type DragEvent } from "react";
import { Upload, X, FileText, Sparkles, ArrowRight } from "lucide-react";

interface ResumeInputProps {
  onAnalyze: (text: string) => void;
  disabled: boolean;
}

export function ResumeInput({ onAnalyze, disabled }: ResumeInputProps) {
  const [resumeText, setResumeText] = useState("");
  const [fileName, setFileName] = useState("");
  const [fileError, setFileError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setFileError(null);
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setFileError(`"${file.name}" is not a .txt file. Please upload plain text (.txt) files only.`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = (e.target?.result as string) || "";
      if (!text.trim()) {
        setFileError(`"${file.name}" is empty. Please upload a file with resume content.`);
        return;
      }
      setResumeText(text);
      setFileName(file.name);
      setFileError(null);
    };
    reader.onerror = () => {
      setFileError("Error reading file. Please try again.");
    };
    reader.readAsText(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleClear = () => {
    setResumeText("");
    setFileName("");
    setFileError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAnalyze = () => {
    const trimmed = resumeText.trim();
    if (trimmed) {
      onAnalyze(trimmed);
    } else {
      setFileError("Please provide resume text before analyzing.");
    }
  };

  const charCount = resumeText.length;
  const isValid = resumeText.trim().length > 0;  // used for button state and clear btn


  const handleInsertSample = () => {
    setResumeText(
      "Senior Machine Learning Engineer with 5+ years of experience developing deep learning models and NLP pipelines using Python, PyTorch, Scikit-Learn, and Hugging Face. Led deployment of production recommendation systems with FastAPI, Redis, and Kubernetes. Strong background in data modeling, feature engineering, and MLOps."
    );
    setFileName("");
    setFileError(null);
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Main Elevated Analyzer Card */}
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-sm transition-all duration-200">
        <div className="border-b border-slate-100 px-6 py-5 sm:px-8">
          <h2 className="text-xl font-bold tracking-tight text-slate-900">
            Analyze Resume
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Upload a TXT resume or paste resume text below.
          </p>
        </div>

        <div className="p-6 sm:p-8">
          {/* Polished Drag & Drop Zone */}
          <div
            className={`relative mb-6 rounded-xl border-2 border-dashed p-6 text-center transition-all duration-200 sm:p-8 ${
              isDragging
                ? "border-blue-500 bg-blue-50/60 ring-4 ring-blue-500/10"
                : "border-slate-300/80 bg-slate-50/50 hover:border-slate-400 hover:bg-slate-50"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
              disabled={disabled}
            />

            {fileName ? (
              <div className="flex flex-wrap items-center justify-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 text-blue-700">
                  <FileText className="h-5 w-5" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-semibold text-slate-900">{fileName}</p>
                  <p className="text-xs text-slate-500">
                    {charCount.toLocaleString()} characters loaded
                  </p>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClear();
                  }}
                  disabled={disabled}
                  className="ml-2 rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-700"
                  title="Remove file"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-blue-50 text-blue-600">
                  <Upload className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-800">
                    Drop your resume here
                  </p>
                  <p className="text-xs text-slate-500">
                    TXT files only
                  </p>
                </div>
                <div className="pt-1">
                  <label
                    htmlFor="file-upload"
                    className="inline-flex cursor-pointer items-center justify-center rounded-lg border border-slate-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 shadow-2xs transition-all hover:bg-slate-50 hover:text-slate-900 active:scale-98 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Browse files
                  </label>
                </div>
              </div>
            )}
          </div>

          {fileError && (
            <div className="mb-5 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50/90 px-3.5 py-2.5 text-xs font-medium text-red-700">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              {fileError}
            </div>
          )}

          {/* Professional Textarea */}
          <div className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <label
                htmlFor="resume-text"
                className="text-xs font-semibold uppercase tracking-wider text-slate-600"
              >
                Or paste resume text
              </label>
              <button
                type="button"
                onClick={handleInsertSample}
                disabled={disabled}
                className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 transition-colors hover:text-blue-800 hover:underline disabled:opacity-50"
              >
                <Sparkles className="h-3 w-3" />
                Insert sample resume
              </button>
            </div>
            <textarea
              id="resume-text"
              value={resumeText}
              onChange={(e) => {
                setResumeText(e.target.value);
                if (fileError) setFileError(null);
              }}
              disabled={disabled}
              rows={7}
              className="w-full rounded-xl border border-slate-200 bg-white p-4 font-mono text-xs leading-relaxed text-slate-800 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none focus:ring-3 focus:ring-blue-500/15 disabled:bg-slate-50 disabled:text-slate-400"
              placeholder="Paste raw resume text here (experience, skills, education)..."
            />
          </div>

          {/* Footer Actions */}
          <div className="flex flex-col-reverse items-stretch justify-between gap-3 sm:flex-row sm:items-center">
            <div className="text-xs font-medium text-slate-500">
              {charCount > 0 ? (
                <span className="inline-flex items-center gap-1.5 rounded-md bg-slate-100 px-2.5 py-1 text-slate-700">
                  <span className="font-semibold text-slate-900">{charCount.toLocaleString()}</span> characters
                </span>
              ) : (
                <span>Ready to analyze</span>
              )}
            </div>

            <div className="flex items-center gap-2.5">
              {isValid && (
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={disabled}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs transition-all hover:bg-slate-50 active:scale-98 disabled:opacity-50"
                >
                  <X className="h-3.5 w-3.5" /> Clear
                </button>
              )}
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!isValid || disabled}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-xs transition-all hover:bg-blue-700 active:scale-98 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
              >
                Analyze Resume
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

