### Adaptive Model Engine ###

Adaptive Model Engine is a runtime hardware-aware model selector designed for local AI inference in the browser.

It dynamically selects the most appropriate Whisper transcription model and backend based on the user's CPU, RAM, GPU capabilities, and device type.

The engine automatically decides between:

WebGPU ONNX inference

CPU inference using whisper.cpp

while selecting the largest stable model that the system can run reliably.

The goal is to provide maximum transcription quality without causing crashes, memory exhaustion, or unusable performance.

This engine powers Cowslator, enabling fully local transcription directly in the browser.

Design Goals

The adaptive engine was designed to solve a common problem in browser AI applications:

Users have extremely heterogeneous hardware.

#Devices range from:

low-end mobile phones

laptops without GPUs

desktops with powerful GPUs

Loading the wrong model can result in:

out-of-memory crashes

unusable latency

GPU allocation failures

browser tab termination

#The Adaptive Model Engine solves this through:

hardware probing

WebGPU capability testing

memory estimation

tier-based model selection

safe fallback strategies

### Architecture Overview ###

Client Hardware
      │
      │
      ▼
Device Detection (PHP + UA)
      │
      ▼
GPU Capability Probe (WebGPU)
      │
      ▼
Hardware Scoring System
      │
      ▼
Tier Selection
      │
      ▼
Backend Selection
  ├─ ONNX WebGPU
  └─ whisper.cpp CPU
      │
      ▼
Model Selection
      │
      ▼
Memory Estimation
      │
      ▼
Confidence & Fallback

### Hardware Detection Pipeline ###

The engine gathers system information from multiple sources.

CPU Threads
navigator.hardwareConcurrency

Used to determine parallelism capacity.

## RAM ##

Provided externally (for example via server hints or configuration):

ramGB
Device Type

Determined using server-side user agent analysis.

device-detect.php

$mobilePattern = '/android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile|windows phone/';

Returns:

{
  "deviceType": "mobile"
}

This avoids relying solely on unreliable browser heuristics.

## WebGPU Detection ##

The engine checks whether WebGPU is available:

navigator.gpu.requestAdapter()

If unavailable, the system automatically falls back to CPU inference.

### GPU Allocation Probe ###

Detecting GPU presence alone is insufficient.

Many systems expose WebGPU but fail during large memory allocations.

To solve this, the engine performs progressive GPU allocation tests.

Probe Steps
GPU_TEST_STEPS_MB = [256, 512, 1024, 2048]

Each step attempts to allocate a storage buffer.

Example:

device.createBuffer({
    size: requestedBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
})

If allocation fails, probing stops.

The engine records:

maximum successful allocation

adapter limits

failure reasons

Example result:

maxAllocationBytes = 1024MB

This determines whether GPU inference is viable.

Hardware Scoring System

A weighted scoring algorithm evaluates the device capability.

score = (threads * 2) + (ramGB * 3)

Additional adjustments:

Condition	Score Change
WebGPU available	+10
Large GPU allocation	+20
Mobile device	−10

This produces a hardware capability score.

Example:

threads = 16
ramGB = 32

score = (16*2) + (32*3) + 10 + 20
score = 32 + 96 + 30
score = 158
Tier Mapping

The score is mapped to a model tier.

Score	Tier
<15	tiny
<30	base
<45	small
<60	medium
<75	large-turbo
≥75	large

These tiers represent increasing model sizes.

### Model Library ###

Each model contains metadata describing:

RAM requirements

thread requirements

approximate model size

transcription speed

transcription accuracy

Example:

large-turbo-q8_0
size: 875MB
minRam: 16GB
threads: 12
precision: 5
speed: 2
Practical Fit Model Map

Even if the score suggests a large model, hardware limits may prevent running it.

The engine therefore uses a practical fit table:

MODEL_MAP

Example:

medium
  GPU:
      RAM >= 8GB
      threads >= 8
      GPU allocation >= 4096MB

This prevents selecting models that would crash during inference.

### Backend Selection ###

The engine selects between two backends.

WebGPU Backend
backend = "onnx-webgpu"

Requirements:

WebGPU available

GPU allocation probe successful

hardware meets MODEL_MAP GPU requirements

This enables GPU-accelerated inference.

CPU Backend
backend = "whisper.cpp"

Used when:

WebGPU unavailable

GPU memory insufficient

device unstable

CPU mode uses quantized models optimized for whisper.cpp.

### GPU-First Policy ###

If GPU capabilities exceed the score-based tier:

GPU tier overrides score tier

Example:

scoreTier = small
gpuFitTier = medium

selectedTier = medium

This ensures GPU resources are fully utilized.

### Model Selection ###

Once tier and backend are known:

GPU Models
ONNX_TIER_MODEL

Example:

medium → medium-q5_0
CPU Models
CPU_TIER_MODEL

Example:

medium → medium-q5_0
Memory Estimation

Memory usage is estimated using a multiplier over model size.

onnx-webgpu factor = 2.3
cpu factor = 1.7

Additional adjustments:

Model	Adjustment
q8_0	+0.25
q4_0	−0.2

Example:

medium model size = 515MB
GPU factor = 2.3

estimated memory ≈ 1184MB

This helps avoid memory overcommit.

### Confidence Estimation ###

Each decision returns a confidence value.

high
medium
low

Confidence depends on:

distance between score and tier thresholds

device type

fallback triggers

Mobile devices near thresholds reduce confidence.

### Safe Fallback System ###

If evaluation fails or produces unsafe conditions, the engine triggers a safe fallback.

tiny-q4_0
backend = whisper.cpp

This guarantees transcription remains functional even on extremely weak devices.

Decision Output

The evaluate() function returns a full decision object.

Example:

{
  "tier": "large-turbo",
  "selectedModel": "large-turbo-q8_0",
  "selectedBackend": "onnx-webgpu",
  "estimatedMemoryMb": 2231,
  "confidence": "high",
  "score": 138,
  "scoreTier": "large",
  "practicalTierCap": "large-turbo",
  "hardware": {
    "threads": 20,
    "ramGB": 32,
    "deviceType": "desktop",
    "hasWebGPU": true
  }
}
Logging and Debugging

The engine emits structured logs:

[adaptive] hardware summary
[adaptive] gpu test result
[adaptive] selected model tier

This enables detailed diagnostics when tuning model selection.

### Public API ###

The engine exposes a small API.

Evaluate Hardware
AdaptiveModelEngine.evaluate(options)

Returns a full decision object.

Retrieve Model Metadata
AdaptiveModelEngine.getModelMeta(modelId)
Safe Fallback
AdaptiveModelEngine.buildSafeFallbackDecision(reason)
Model Library
AdaptiveModelEngine.library
Tier Order
AdaptiveModelEngine.tiers
Why This Matters

Most browser AI applications force users to manually choose models.

This engine provides:

automatic hardware adaptation

crash prevention

GPU exploitation when available

stable CPU fallback

deterministic behavior across devices

It enables production-grade local AI inference directly in the browser.

Use Case: Cowslator

The Adaptive Model Engine is used in Cowslator to dynamically choose the best Whisper model for transcription workloads.

This allows users to run:

local audio transcription

video subtitle generation

batch transcription

without needing to configure model sizes manually.

### License ###

MIT License
