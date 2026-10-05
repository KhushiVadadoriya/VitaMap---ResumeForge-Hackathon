export type AppState = "idle" | "processing" | "success" | "error";

export interface PredictRequest {
  resume_text: string;
}

export interface PredictResponse {
  category: string;
  confidence: string | number;
  model: string;
  top_terms: string[];
}

export interface ModelMetric {
  name: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1?: number;
}

export interface MetricsResponse {
  models: ModelMetric[];
  confusion_matrix: number[][];
  labels: string[];
}

export type ProcessingStage =
  | "resume_received"
  | "text_preprocessing"
  | "feature_extraction"
  | "model_inference"
  | "category_identified";
