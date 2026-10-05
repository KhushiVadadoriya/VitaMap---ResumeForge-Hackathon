import type { PredictResponse, MetricsResponse } from "./types";

export const MOCK_PREDICTION: PredictResponse = {
  category: "Data Science",
  confidence: 0.91,
  model: "TF-IDF + Linear SVM",
  top_terms: ["Python", "Machine Learning", "SQL", "Pandas", "TensorFlow"],
};

export const MOCK_METRICS: MetricsResponse = {
  models: [
    {
      name: "TF-IDF + Linear SVM",
      accuracy: 0.94,
      macro_f1: 0.93,
      weighted_f1: 0.94,
    },
    {
      name: "Word2Vec + LSTM",
      accuracy: 0.91,
      macro_f1: 0.9,
      weighted_f1: 0.91,
    },
  ],
  confusion_matrix: [
    [42, 1, 0, 2],
    [2, 38, 1, 0],
    [0, 2, 35, 1],
    [1, 0, 2, 40],
  ],
  labels: ["Data Science", "HR", "Web Development", "Engineering"],
};
