(function (global) {
  'use strict';

  var MB = 1024 * 1024;
  var GB = 1024 * MB;
  var GPU_TEST_STEPS_MB = [256, 512, 1024, 2048];
  var probeCache = null;
  var probePromise = null;

  var MODEL_LIBRARY = {
    'tiny-q4_0': {
      id: 'tiny-q4_0',
      tier: 'tiny',
      sizeMb: 23,
      minRam: 1,
      minThreads: 1,
      precision: 1,
      speed: 5,
      file: '/model/ggml-tiny-q4_0.bin'
    },
    'tiny-q5_1': {
      id: 'tiny-q5_1',
      tier: 'tiny',
      sizeMb: 31,
      minRam: 2,
      minThreads: 2,
      precision: 2,
      speed: 5,
      file: '/model/ggml-tiny-q5_1.bin'
    },
    'base-q5_1': {
      id: 'base-q5_1',
      tier: 'base',
      sizeMb: 57,
      minRam: 4,
      minThreads: 4,
      precision: 3,
      speed: 4,
      file: '/model/ggml-base-q5_1.bin'
    },
    'small-q5_1': {
      id: 'small-q5_1',
      tier: 'small',
      sizeMb: 182,
      minRam: 8,
      minThreads: 6,
      precision: 4,
      speed: 3,
      file: '/model/ggml-small-q5_1.bin'
    },
    'medium-q5_0': {
      id: 'medium-q5_0',
      tier: 'medium',
      sizeMb: 515,
      minRam: 12,
      minThreads: 8,
      precision: 4,
      speed: 2,
      file: '/model/ggml-medium-q5_0.bin'
    },
    'large-turbo-q8_0': {
      id: 'large-turbo-q8_0',
      tier: 'large-turbo',
      sizeMb: 875,
      minRam: 16,
      minThreads: 12,
      precision: 5,
      speed: 2,
      file: '/model/ggml-large-v3-turbo-q8_0.bin'
    },
    'large-q5_0': {
      id: 'large-q5_0',
      tier: 'large',
      sizeMb: 1030,
      minRam: 16,
      minThreads: 12,
      precision: 5,
      speed: 1,
      file: '/model/ggml-large-v3-q5_0.bin'
    }
  };

  var TIER_ORDER = ['tiny', 'base', 'small', 'medium', 'large-turbo', 'large'];
  // Practical fit table aligned with model_map.txt.
  var MODEL_MAP = {
    tiny: {
      onnxGpu: { ramGB: 1, threads: 2, gpuMb: 512 },
      cpuQ51: { ramGB: 1, threads: 2 },
      cpuQ8: { ramGB: 1.5, threads: 2 }
    },
    base: {
      onnxGpu: { ramGB: 2, threads: 4, gpuMb: 1024 },
      cpuQ51: { ramGB: 2, threads: 4 },
      cpuQ8: { ramGB: 3, threads: 4 }
    },
    small: {
      onnxGpu: { ramGB: 4, threads: 6, gpuMb: 2048 },
      cpuQ51: { ramGB: 4, threads: 6 },
      cpuQ8: { ramGB: 6, threads: 8 }
    },
    medium: {
      onnxGpu: { ramGB: 8, threads: 8, gpuMb: 4096 },
      cpuQ51: { ramGB: 8, threads: 8 },
      cpuQ8: { ramGB: 10, threads: 12 }
    },
    'large-turbo': {
      onnxGpu: { ramGB: 12, threads: 12, gpuMb: 6144 },
      cpuQ51: { ramGB: 12, threads: 12 },
      cpuQ8: { ramGB: 16, threads: 16 }
    },
    large: {
      onnxGpu: { ramGB: 16, threads: 16, gpuMb: 8192 },
      cpuQ51: { ramGB: 16, threads: 16 },
      cpuQ8: { ramGB: 20, threads: 20 }
    }
  };
  var ONNX_TIER_MODEL = {
    tiny: 'tiny-q5_1',
    base: 'base-q5_1',
    small: 'small-q5_1',
    medium: 'medium-q5_0',
    'large-turbo': 'large-turbo-q8_0',
    large: 'large-q5_0'
  };
  var CPU_TIER_MODEL = {
    tiny: 'tiny-q5_1',
    base: 'base-q5_1',
    small: 'small-q5_1',
    medium: 'medium-q5_0',
    'large-turbo': 'large-turbo-q8_0',
    large: 'large-q5_0'
  };

  function toInt(value, fallback) {
    var n = parseInt(value, 10);
    return isFinite(n) ? n : fallback;
  }

  function clampMin(value, min) {
    if (!isFinite(value)) return min;
    return value < min ? min : value;
  }

  function tierIndex(tier) {
    var idx = TIER_ORDER.indexOf(tier);
    return idx === -1 ? 0 : idx;
  }

  function minTierByOrder(a, b) {
    return tierIndex(a) <= tierIndex(b) ? a : b;
  }

  function detectClientDeviceType() {
    var ua = '';
    if (global.navigator && global.navigator.userAgent) {
      ua = String(global.navigator.userAgent).toLowerCase();
    }
    var isMobile = /android|iphone|ipad|ipod|mobile|windows phone|opera mini|iemobile/.test(ua);
    return isMobile ? 'mobile' : 'desktop';
  }

  async function detectDeviceTypeFromServer(options) {
    var endpoint = options && options.deviceEndpoint ? options.deviceEndpoint : '/device-detect.php';
    try {
      var response = await fetch(endpoint, {
        method: 'GET',
        cache: 'no-store',
        headers: { Accept: 'application/json' }
      });
      if (!response.ok) {
        throw new Error('HTTP ' + response.status);
      }
      var data = await response.json();
      var type = String((data && data.deviceType) || '').toLowerCase();
      if (type !== 'mobile' && type !== 'desktop') {
        throw new Error('Invalid deviceType from server');
      }
      return {
        deviceType: type,
        source: 'php'
      };
    } catch (err) {
      return {
        deviceType: detectClientDeviceType(),
        source: 'client-fallback',
        error: String(err && err.message ? err.message : err)
      };
    }
  }

  function normalizeAdapterLimits(limits) {
    if (!limits) return {};
    var out = {};
    var keys = [
      'maxBufferSize',
      'maxStorageBufferBindingSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupSizeZ'
    ];
    for (var i = 0; i < keys.length; i++) {
      var key = keys[i];
      if (typeof limits[key] !== 'undefined') {
        out[key] = Number(limits[key]);
      }
    }
    return out;
  }

  async function runGpuAllocationProbe(adapter) {
    var result = {
      hasWebGPU: false,
      passed: false,
      maxAllocationBytes: 0,
      adapterLimits: {},
      steps: [],
      error: ''
    };

    if (!adapter) {
      result.error = 'No WebGPU adapter available';
      return result;
    }

    result.hasWebGPU = true;
    result.adapterLimits = normalizeAdapterLimits(adapter.limits);

    var device = null;
    try {
      device = await adapter.requestDevice();
      var maxBufferLimit = result.adapterLimits.maxBufferSize || Number.MAX_SAFE_INTEGER;
      for (var i = 0; i < GPU_TEST_STEPS_MB.length; i++) {
        var stepMb = GPU_TEST_STEPS_MB[i];
        var requestedBytes = stepMb * MB;
        if (requestedBytes > maxBufferLimit) {
          result.steps.push({
            sizeMb: stepMb,
            ok: false,
            reason: 'exceeds adapter maxBufferSize'
          });
          break;
        }

        try {
          // Try lightweight STORAGE buffer allocations to estimate usable VRAM budget.
          var buf = device.createBuffer({
            size: requestedBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          });
          if (buf && typeof buf.destroy === 'function') {
            buf.destroy();
          }
          result.maxAllocationBytes = requestedBytes;
          result.steps.push({
            sizeMb: stepMb,
            ok: true
          });
        } catch (stepErr) {
          result.steps.push({
            sizeMb: stepMb,
            ok: false,
            reason: String(stepErr && stepErr.message ? stepErr.message : stepErr)
          });
          break;
        }
      }
      result.passed = result.maxAllocationBytes >= (256 * MB);
    } catch (err) {
      result.error = String(err && err.message ? err.message : err);
      result.passed = false;
    } finally {
      if (device && typeof device.destroy === 'function') {
        try {
          device.destroy();
        } catch (ignore) {}
      }
    }

    return result;
  }

  async function detectGpuHardware() {
    if (!global.navigator || !global.navigator.gpu) {
      return {
        hasWebGPU: false,
        passed: false,
        maxAllocationBytes: 0,
        adapterLimits: {},
        steps: [],
        error: 'navigator.gpu is unavailable'
      };
    }

    try {
      var adapter = await global.navigator.gpu.requestAdapter();
      if (!adapter) {
        return {
          hasWebGPU: false,
          passed: false,
          maxAllocationBytes: 0,
          adapterLimits: {},
          steps: [],
          error: 'requestAdapter() returned null'
        };
      }
      return await runGpuAllocationProbe(adapter);
    } catch (err) {
      return {
        hasWebGPU: false,
        passed: false,
        maxAllocationBytes: 0,
        adapterLimits: {},
        steps: [],
        error: String(err && err.message ? err.message : err)
      };
    }
  }

  async function getProbeSnapshot(options) {
    if (options && options.forceRefreshProbe) {
      probeCache = null;
      probePromise = null;
    }
    if (probeCache) {
      return probeCache;
    }
    if (probePromise) {
      return probePromise;
    }

    probePromise = (async function () {
      var deviceInfo = await detectDeviceTypeFromServer(options || {});
      var gpuTest = await detectGpuHardware();
      probeCache = {
        deviceInfo: deviceInfo,
        gpuTest: gpuTest
      };
      probePromise = null;
      return probeCache;
    })();

    return probePromise;
  }

  function scoreHardware(hardware) {
    var score = (hardware.threads * 2) + (hardware.ramGB * 3);
    if (hardware.hasWebGPU) {
      score += 10;
    }
    if (
      hardware.maxAllocationBytes > (2 * GB) ||
      (hardware.maxAllocationBytes >= (2 * GB) &&
        hardware.adapterLimits &&
        Number(hardware.adapterLimits.maxBufferSize || 0) > (2 * GB))
    ) {
      score += 20;
    }
    if (hardware.deviceType === 'mobile') {
      score -= 10;
    }
    return score;
  }

  function mapScoreToTier(score) {
    if (score < 15) return 'tiny';
    if (score < 30) return 'base';
    if (score < 45) return 'small';
    if (score < 60) return 'medium';
    if (score < 75) return 'large-turbo';
    return 'large';
  }

  function getGpuAllocationMb(hardware) {
    return Math.round((hardware.maxAllocationBytes || 0) / MB);
  }

  function getGpuThreadRequirement(baseThreads, hardware) {
    var requirement = baseThreads;
    if (hardware.deviceType !== 'mobile') {
      // For GPU inference, CPU threads are still relevant, but less strict than pure CPU mode.
      requirement = Math.max(2, baseThreads - 2);
    }
    if (getGpuAllocationMb(hardware) >= 4096) {
      requirement = Math.max(2, requirement - 1);
    }
    return requirement;
  }

  function meetsOnnxTierRequirements(tier, hardware) {
    var req = MODEL_MAP[tier] && MODEL_MAP[tier].onnxGpu;
    if (!req) return false;
    if (!hardware.hasWebGPU || !hardware.gpuAllocationPassed) return false;
    if (hardware.ramGB < req.ramGB) return false;
    if (hardware.threads < getGpuThreadRequirement(req.threads, hardware)) return false;
    if (getGpuAllocationMb(hardware) < req.gpuMb) return false;
    return true;
  }

  function meetsCpuQ51TierRequirements(tier, hardware) {
    var req = MODEL_MAP[tier] && MODEL_MAP[tier].cpuQ51;
    if (!req) return false;
    if (hardware.ramGB < req.ramGB) return false;
    if (hardware.threads < req.threads) return false;
    return true;
  }

  function getHighestFittingTier(hardware, mode) {
    for (var i = TIER_ORDER.length - 1; i >= 0; i--) {
      var tier = TIER_ORDER[i];
      var ok = mode === 'onnx-webgpu'
        ? meetsOnnxTierRequirements(tier, hardware)
        : meetsCpuQ51TierRequirements(tier, hardware);
      if (ok) return tier;
    }
    return 'tiny';
  }

  function pickWhisperFallbackModel(tier, hardware) {
    if (hardware.ramGB < 2) {
      return {
        modelId: 'tiny-q4_0',
        reason: 'RAM below 2GB'
      };
    }

    // Guardrail: keep ultra-low CPU profiles on tiny only.
    if (hardware.threads <= 2 || hardware.cores <= 1) {
      return {
        modelId: 'tiny-q5_1',
        reason: 'Ultra-low CPU profile: restricting to tiny for stability'
      };
    }
    // Mobile devices are usually less stable with sustained CPU transcription.
    if (hardware.deviceType === 'mobile' && (hardware.threads <= 4 || hardware.cores <= 2)) {
      return {
        modelId: 'tiny-q5_1',
        reason: 'Mobile low-core CPU profile: restricting to tiny for stability'
      };
    }

    if (hardware.ramGB > 12 && hardware.threads > 12 && (tier === 'large-turbo' || tier === 'large')) {
      return {
        modelId: 'large-turbo-q8_0',
        reason: 'High RAM/threads: allowing q8_0'
      };
    }

    return {
      modelId: CPU_TIER_MODEL[tier] || 'tiny-q5_1',
      reason: 'CPU fallback constrained by model_map practical fit'
    };
  }

  function pickOnnxModelForTier(tier) {
    return ONNX_TIER_MODEL[tier] || 'tiny-q5_1';
  }

  function estimateMemoryUsageMb(modelId, backend) {
    var meta = MODEL_LIBRARY[modelId] || MODEL_LIBRARY['tiny-q4_0'];
    var factor = backend === 'onnx-webgpu' ? 2.3 : 1.7;
    if (modelId.indexOf('q8_0') !== -1) factor += 0.25;
    if (modelId.indexOf('q4_0') !== -1) factor -= 0.2;
    return Math.max(120, Math.round(meta.sizeMb * factor));
  }

  function computeConfidence(score, tier, hardware, fallbackReason) {
    if (fallbackReason && fallbackReason.indexOf('safe fallback') !== -1) {
      return 'low';
    }

    var threshold = 0;
    switch (tier) {
      case 'tiny': threshold = 15; break;
      case 'base': threshold = 30; break;
      case 'small': threshold = 45; break;
      case 'medium': threshold = 60; break;
      case 'large-turbo': threshold = 75; break;
      default: threshold = 90; break;
    }
    var distance = Math.abs(threshold - score);
    if (hardware.deviceType === 'mobile' && distance < 8) {
      return 'low';
    }
    if (distance >= 10) return 'high';
    if (distance >= 5) return 'medium';
    return 'low';
  }

  function buildSafeFallbackDecision(reason, base) {
    var message = reason ? String(reason) : 'Unknown adaptive-engine failure';
    var hardware = base && base.hardware ? base.hardware : {
      threads: 1,
      ramGB: 1,
      hasWebGPU: false,
      maxAllocationBytes: 0,
      adapterLimits: {},
      deviceType: 'desktop'
    };
    return {
      hardware: hardware,
      score: 0,
      scoreTier: 'tiny',
      practicalTierCap: 'tiny',
      tier: 'tiny',
      selectedModel: 'tiny-q4_0',
      selectedBackend: 'whisper.cpp',
      estimatedMemoryMb: estimateMemoryUsageMb('tiny-q4_0', 'whisper.cpp'),
      confidence: 'low',
      fallbackReason: 'safe fallback triggered: ' + message,
      gpuTest: base && base.gpuTest ? base.gpuTest : {
        hasWebGPU: false,
        passed: false,
        maxAllocationBytes: 0,
        adapterLimits: {},
        steps: [],
        error: ''
      },
      recommendedModelIds: ['tiny-q4_0', 'tiny-q5_1']
    };
  }

  function buildRecommendedModelIds(selectedTier, selectedBackend) {
    var ids = [];
    var map = selectedBackend === 'onnx-webgpu' ? ONNX_TIER_MODEL : CPU_TIER_MODEL;
    var start = tierIndex(selectedTier);
    for (var i = start; i >= 0; i--) {
      var tier = TIER_ORDER[i];
      var modelId = map[tier];
      if (modelId && ids.indexOf(modelId) === -1) {
        ids.push(modelId);
      }
    }
    if (ids.indexOf('tiny-q5_1') === -1) ids.push('tiny-q5_1');
    if (ids.indexOf('tiny-q4_0') === -1) ids.push('tiny-q4_0');
    return ids;
  }

  async function evaluate(options) {
    var opts = options || {};
    var threads = clampMin(toInt(opts.threads, (global.navigator && global.navigator.hardwareConcurrency) || 4), 1);
    var cores = clampMin(toInt(opts.cores, Math.max(1, Math.floor(threads / 2))), 1);
    var ramGB = clampMin(toInt(opts.ramGB, 4), 1);

    var probe = await getProbeSnapshot(opts);
    var deviceInfo = probe.deviceInfo;
    var gpuTest = probe.gpuTest;

    var hardware = {
      cores: cores,
      threads: threads,
      ramGB: ramGB,
      hasWebGPU: Boolean(gpuTest.hasWebGPU),
      maxAllocationBytes: gpuTest.maxAllocationBytes || 0,
      adapterLimits: gpuTest.adapterLimits || {},
      gpuAllocationPassed: Boolean(gpuTest.passed),
      deviceType: deviceInfo.deviceType || 'desktop',
      deviceDetectionSource: deviceInfo.source || 'unknown'
    };

    var score = scoreHardware(hardware);
    var scoreTier = mapScoreToTier(score);
    var tier = scoreTier;
    var selectedBackend = 'whisper.cpp';
    var fallbackReason = '';
    var selectedModel = '';
    var practicalTierCap = 'tiny';

    if (hardware.hasWebGPU && hardware.gpuAllocationPassed) {
      practicalTierCap = getHighestFittingTier(hardware, 'onnx-webgpu');
      selectedBackend = 'onnx-webgpu';
      // GPU-first policy: when GPU practical fit is stronger than score tier, favor GPU fit.
      tier = practicalTierCap;
      selectedModel = pickOnnxModelForTier(tier);
      if (tierIndex(practicalTierCap) < tierIndex(scoreTier)) {
        fallbackReason = 'Score tier "' + scoreTier + '" capped to "' + tier + '" using model_map practical GPU fit.';
      } else if (tierIndex(practicalTierCap) > tierIndex(scoreTier)) {
        fallbackReason = 'GPU-first policy elevated score tier "' + scoreTier + '" to "' + tier + '" using model_map practical GPU fit.';
      }
    } else {
      practicalTierCap = getHighestFittingTier(hardware, 'whisper.cpp');
      selectedBackend = 'whisper.cpp';
      tier = minTierByOrder(scoreTier, practicalTierCap);
      var whisperPick = pickWhisperFallbackModel(tier, hardware);
      selectedModel = whisperPick.modelId;
      fallbackReason = whisperPick.reason;
      if (tier !== scoreTier) {
        fallbackReason = 'Score tier "' + scoreTier + '" capped to "' + tier + '" using model_map practical CPU fit. ' + fallbackReason;
      }
    }

    var estimatedMemoryMb = estimateMemoryUsageMb(selectedModel, selectedBackend);
    var confidence = computeConfidence(score, tier, hardware, fallbackReason);

    var decision = {
      hardware: hardware,
      score: score,
      tier: tier,
      selectedModel: selectedModel,
      selectedBackend: selectedBackend,
      estimatedMemoryMb: estimatedMemoryMb,
      confidence: confidence,
      fallbackReason: fallbackReason,
      scoreTier: scoreTier,
      practicalTierCap: practicalTierCap,
      gpuTest: gpuTest,
      recommendedModelIds: buildRecommendedModelIds(tier, selectedBackend)
    };

    console.info('[adaptive] hardware summary', {
      threads: hardware.threads,
      cores: hardware.cores,
      ramGB: hardware.ramGB,
      deviceType: hardware.deviceType,
      hasWebGPU: hardware.hasWebGPU,
      gpuAllocationPassed: hardware.gpuAllocationPassed
    });
    console.info('[adaptive] gpu test result', {
      passed: gpuTest.passed,
      maxAllocationMB: Math.round((gpuTest.maxAllocationBytes || 0) / MB),
      adapterLimits: gpuTest.adapterLimits,
      steps: gpuTest.steps,
      error: gpuTest.error || ''
    });
    console.info('[adaptive] selected model tier', {
      score: decision.score,
      scoreTier: decision.scoreTier,
      practicalTierCap: decision.practicalTierCap,
      selectedTier: decision.tier,
      selectedModel: decision.selectedModel,
      selectedBackend: decision.selectedBackend,
      estimatedMemoryMb: decision.estimatedMemoryMb,
      confidence: decision.confidence
    });
    if (decision.fallbackReason) {
      console.warn('[adaptive] fallback reason', decision.fallbackReason);
    }
    if (deviceInfo.error) {
      console.warn('[adaptive] device detection fallback', deviceInfo.error);
    }

    return decision;
  }

  function getModelMeta(modelId) {
    if (MODEL_LIBRARY[modelId]) {
      return MODEL_LIBRARY[modelId];
    }
    return MODEL_LIBRARY['tiny-q4_0'];
  }

  global.AdaptiveModelEngine = {
    evaluate: evaluate,
    getModelMeta: getModelMeta,
    buildSafeFallbackDecision: buildSafeFallbackDecision,
    library: MODEL_LIBRARY,
    tiers: TIER_ORDER
  };
})(window);
