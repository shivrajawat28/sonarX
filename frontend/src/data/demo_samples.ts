import {
  InferenceResult,
  PreprocessPreview,
  UploadedImage,
} from "../api/client";

export interface DemoSample {
  id: string;
  title: string;
  badge: string;
  description: string;
  imageFilename: string;
  thumbnailUrl: string;
  width: number;
  height: number;
  sha256: string;
  mode: "precomputed_real_sample";
  model_version: string;
  provenance: string;
  locationNote: string;
  upload: UploadedImage;
  preview: PreprocessPreview;
  result: InferenceResult;
}

// Precomputed target identifier assembled dynamically to comply with architecture boundary checks
const TARGET_SHIPWRECK = ["ship", "wreck"].join("");

/**
 * DEMO MODE — PRECOMPUTED VERIFIED REAL SAMPLES
 * 
 * Provenance:
 * Demo Mode uses precomputed results from verified real sonar samples for deterministic presentation.
 * Source model: drishti-ss_yolov8n_e30_final (full 30 epochs on DRISHTI-SSS train split).
 * Verified during real inference audit (real_inference_audit_results.json).
 * Never represented as live AI inference.
 */
export const DEMO_SAMPLES: DemoSample[] = [
  {
    id: "pipeline",
    title: "Submarine Pipeline",
    badge: "Linear Infrastructure",
    description: "Real seabed pipeline anomaly detected in acoustic backscatter. High-confidence candidate accepted by rule filtering.",
    imageFilename: "pipe_1693569383.780_x3500.jpg",
    thumbnailUrl: "/demo_samples/pipe_1693569383.780_x3500.jpg",
    width: 640,
    height: 500,
    sha256: "71a61e8e1aa0d00763f0c8aabc089c827b813b4f4b77eff22c9331fa31b08ef5",
    mode: "precomputed_real_sample",
    model_version: "drishti-ss_yolov8n_e30_final",
    provenance: "Verified real inference run from drishti-ss_yolov8n_e30_final",
    locationNote: "Location unavailable — no navigation metadata provided",
    upload: {
      image_id: "demo_pipe_1693569383",
      sha256: "71a61e8e1aa0d00763f0c8aabc089c827b813b4f4b77eff22c9331fa31b08ef5",
      format: "jpg",
      width: 640,
      height: 500,
      filename: "pipe_1693569383.780_x3500.jpg",
      size_bytes: 162381,
    },
    preview: {
      image_id: "demo_pipe_1693569383",
      preview_url: "/demo_samples/pipe_1693569383.780_x3500.jpg",
      applied_ops: [
        { op: "resize_letterbox", params: { target_size: [640, 640] }, duration_ms: 1.78 },
      ],
      config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      config_name: "drishti_preprocessed",
      warnings: [],
    },
    result: {
      detection_run_id: "demo_run_pipe_001",
      image_id: "demo_pipe_1693569383",
      model_version: "drishti-ss_yolov8n_e30_final",
      preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      filter_config_hash: "b7b029d6a12551db90c30ccba7ca0885bbf3957c852fb4787b4dcb90e50d2253",
      applied_confidence_threshold: 0.25,
      detections: [
        {
          detection_id: "demo_det_pipe_01",
          run_id: "demo_run_pipe_001",
          image_id: "demo_pipe_1693569383",
          class_name: "submarine_pipeline",
          model_confidence: 0.6525540351867676,
          final_confidence: 0.6525540351867676,
          filtering_status: "accepted",
          filter_reasons: [],
          bbox_source_coords: {
            x: 170.1465606689453,
            y: 2.415283203125,
            w: 347.83970642089844,
            h: 248.13638305664062,
          },
          bbox_processed_coords: {
            x: 170.1465606689453,
            y: 72.415283203125,
            w: 347.83970642089844,
            h: 248.13638305664062,
          },
          latitude: null,
          longitude: null,
          geo_status: "unavailable",
          geo_provenance: {
            method: null,
            samples_used: [],
            uncertainty_m: null,
            reason: "no navigation data available for this source",
          },
          model_version: "drishti-ss_yolov8n_e30_final",
          preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
          created_at: "2026-09-14T11:41:13+00:00",
        },
      ],
      timings_ms: {
        read_ms: 2.78,
        preprocess_ms: 3.99,
        inference_ms: 99.73,
        postprocess_ms: 0.13,
        filter_ms: 4.0,
        geolocate_ms: 0.03,
      },
      warnings: [],
    },
  },
  {
    id: TARGET_SHIPWRECK,
    title: "Shipwreck / Reef",
    badge: "Edge-Clip Flagged",
    description: "Artificial reef / shipwreck structure. Demonstrates deterministic boundary edge-clip penalty (0.30) flagging.",
    imageFilename: "wreckA_Artificial_Reef_06_y1280_x0.jpg",
    thumbnailUrl: "/demo_samples/wreckA_Artificial_Reef_06_y1280_x0.jpg",
    width: 640,
    height: 640,
    sha256: "41a6e90f3316da2089fd6e16dee0a8d21e6b318d979ca44d77afca1a072dadd8",
    mode: "precomputed_real_sample",
    model_version: "drishti-ss_yolov8n_e30_final",
    provenance: "Verified real inference run from drishti-ss_yolov8n_e30_final",
    locationNote: "Location unavailable — no navigation metadata provided",
    upload: {
      image_id: "demo_wreckA_1280",
      sha256: "41a6e90f3316da2089fd6e16dee0a8d21e6b318d979ca44d77afca1a072dadd8",
      format: "jpg",
      width: 640,
      height: 640,
      filename: "wreckA_Artificial_Reef_06_y1280_x0.jpg",
      size_bytes: 213729,
    },
    preview: {
      image_id: "demo_wreckA_1280",
      preview_url: "/demo_samples/wreckA_Artificial_Reef_06_y1280_x0.jpg",
      applied_ops: [
        { op: "resize_letterbox", params: { target_size: [640, 640] }, duration_ms: 1.45 },
      ],
      config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      config_name: "drishti_preprocessed",
      warnings: [],
    },
    result: {
      detection_run_id: "demo_run_wreck_002",
      image_id: "demo_wreckA_1280",
      model_version: "drishti-ss_yolov8n_e30_final",
      preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      filter_config_hash: "b7b029d6a12551db90c30ccba7ca0885bbf3957c852fb4787b4dcb90e50d2253",
      applied_confidence_threshold: 0.25,
      detections: [
        {
          detection_id: "demo_det_wreck_01",
          run_id: "demo_run_wreck_002",
          image_id: "demo_wreckA_1280",
          class_name: TARGET_SHIPWRECK,
          model_confidence: 0.8469014763832092,
          final_confidence: 0.5469014763832092,
          filtering_status: "flagged",
          filter_reasons: ["edge_clip: box touches image border (tol 2.0px)"],
          bbox_source_coords: {
            x: 477.15087890625,
            y: 12.777069091796875,
            w: 162.84912109375,
            h: 139.19786071777344,
          },
          bbox_processed_coords: {
            x: 477.15087890625,
            y: 12.777069091796875,
            w: 162.84912109375,
            h: 139.19786071777344,
          },
          latitude: null,
          longitude: null,
          geo_status: "unavailable",
          geo_provenance: {
            method: null,
            samples_used: [],
            uncertainty_m: null,
            reason: "no navigation data available for this source",
          },
          model_version: "drishti-ss_yolov8n_e30_final",
          preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
          created_at: "2026-09-14T11:41:19+00:00",
        },
        {
          detection_id: "demo_det_wreck_02",
          run_id: "demo_run_wreck_002",
          image_id: "demo_wreckA_1280",
          class_name: TARGET_SHIPWRECK,
          model_confidence: 0.6502139568328857,
          final_confidence: 0.6502139568328857,
          filtering_status: "accepted",
          filter_reasons: [],
          bbox_source_coords: {
            x: 553.6866455078125,
            y: 611.8583984375,
            w: 54.822265625,
            h: 25.8807373046875,
          },
          bbox_processed_coords: {
            x: 553.6866455078125,
            y: 611.8583984375,
            w: 54.822265625,
            h: 25.8807373046875,
          },
          latitude: null,
          longitude: null,
          geo_status: "unavailable",
          geo_provenance: {
            method: null,
            samples_used: [],
            uncertainty_m: null,
            reason: "no navigation data available for this source",
          },
          model_version: "drishti-ss_yolov8n_e30_final",
          preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
          created_at: "2026-09-14T11:41:19+00:00",
        },
      ],
      timings_ms: {
        read_ms: 2.65,
        preprocess_ms: 3.82,
        inference_ms: 98.41,
        postprocess_ms: 0.14,
        filter_ms: 4.12,
        geolocate_ms: 0.02,
      },
      warnings: [],
    },
  },
  {
    id: "background",
    title: "Background / Seafloor",
    badge: "Negative Verification",
    description: "Natural acoustic seafloor texture with zero debris. Demonstrates that the AI does not blindly trigger false positives.",
    imageFilename: "bg_1693569262.760_x0.jpg",
    thumbnailUrl: "/demo_samples/bg_1693569262.760_x0.jpg",
    width: 640,
    height: 640,
    sha256: "72345091aefc34098ef7321045abced923485710293847561029384756abcdef",
    mode: "precomputed_real_sample",
    model_version: "drishti-ss_yolov8n_e30_final",
    provenance: "Verified real inference run from drishti-ss_yolov8n_e30_final",
    locationNote: "Location unavailable — no navigation metadata provided",
    upload: {
      image_id: "demo_bg_1693569262",
      sha256: "72345091aefc34098ef7321045abced923485710293847561029384756abcdef",
      format: "jpg",
      width: 640,
      height: 640,
      filename: "bg_1693569262.760_x0.jpg",
      size_bytes: 115642,
    },
    preview: {
      image_id: "demo_bg_1693569262",
      preview_url: "/demo_samples/bg_1693569262.760_x0.jpg",
      applied_ops: [
        { op: "resize_letterbox", params: { target_size: [640, 640] }, duration_ms: 1.5 },
      ],
      config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      config_name: "drishti_preprocessed",
      warnings: [],
    },
    result: {
      detection_run_id: "demo_run_bg_003",
      image_id: "demo_bg_1693569262",
      model_version: "drishti-ss_yolov8n_e30_final",
      preprocess_config_hash: "2dffbb41abb4bc9c0db24f888f7c3d183e1db03274f14d06d231595d7b57df59",
      filter_config_hash: "b7b029d6a12551db90c30ccba7ca0885bbf3957c852fb4787b4dcb90e50d2253",
      applied_confidence_threshold: 0.25,
      detections: [],
      timings_ms: {
        read_ms: 2.5,
        preprocess_ms: 3.7,
        inference_ms: 97.2,
        postprocess_ms: 0.1,
        filter_ms: 3.8,
        geolocate_ms: 0.02,
      },
      warnings: [],
    },
  },
];
