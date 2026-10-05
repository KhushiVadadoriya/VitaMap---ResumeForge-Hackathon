import type { PredictRequest, PredictResponse, MetricsResponse } from "./types";
import { MOCK_PREDICTION, MOCK_METRICS } from "./mockData";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Simulate network delay for mock API
const mockDelay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function isMockMode(): boolean {
  return import.meta.env.VITE_USE_MOCK_API === "true";
}

export async function predictResume(resumeText: string): Promise<PredictResponse> {
  if (isMockMode()) {
    await mockDelay(1200);
    return MOCK_PREDICTION;
  }

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ resume_text: resumeText } as PredictRequest),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      throw new Error(`API error (${response.status}): ${errText || response.statusText}`);
    }

    return response.json();
  } catch (err) {
    if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
      throw new Error(
        `Unable to reach backend API at ${API_URL}. Check that the backend is running, or set VITE_USE_MOCK_API=true.`
      );
    }
    throw err;
  }
}

export async function getMetrics(): Promise<MetricsResponse> {
  if (isMockMode()) {
    await mockDelay(400);
    return MOCK_METRICS;
  }

  try {
    const response = await fetch(`${API_URL}/metrics`);

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      throw new Error(`API error (${response.status}): ${errText || response.statusText}`);
    }

    return response.json();
  } catch (err) {
    if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
      throw new Error(
        `Unable to reach backend API at ${API_URL}. Check that the backend is running, or set VITE_USE_MOCK_API=true.`
      );
    }
    throw err;
  }
}

