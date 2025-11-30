const ACCESS_PASSWORD = "shark";
const ACCESS_KEY = "mathai_access";

const lockScreen = document.getElementById("lock-screen");
const app = document.getElementById("app");
const authForm = document.getElementById("auth-form");
const authPassword = document.getElementById("auth-password");
const authError = document.getElementById("auth-error");
const predictForm = document.getElementById("predict-form");
const buildButton = document.getElementById("build-btn");
const buildStatus = document.getElementById("build-status");
const geometryCountInput = document.getElementById("geometry-count");
const chartCountInput = document.getElementById("chart-count");
const ocrCountInput = document.getElementById("ocr-count");
const lineCountInput = document.getElementById("line-count");
const textCountInput = document.getElementById("text-count");
const trainButton = document.getElementById("train-btn");
const trainEpochsInput = document.getElementById("train-epochs");
const trainStatus = document.getElementById("train-status");
const trainLogs = document.getElementById("train-logs");
const trainProgress = document.getElementById("train-progress");
const trainProgressLabel = document.getElementById("train-progress-label");
const lossPlot = document.getElementById("loss-plot");
const downloadModelBtn = document.getElementById("download-model-btn");
const deleteModelBtn = document.getElementById("delete-model-btn");
const uploadModelBtn = document.getElementById("upload-model-btn");
const uploadModelInput = document.getElementById("model-upload");
const modelStatus = document.getElementById("model-status");
const output = document.getElementById("output");

const unlockApp = () => {
  if (lockScreen) {
    lockScreen.classList.add("hidden");
    lockScreen.style.display = "none";
  }
  if (app) {
    app.classList.remove("hidden");
    app.style.display = "block";
  }
  if (authError) {
    authError.textContent = "";
  }
};

if (sessionStorage.getItem(ACCESS_KEY) === "1") {
  unlockApp();
}

if (authForm) {
  authForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const provided = authPassword.value.trim();
    if (!provided) {
      authError.textContent = "Enter the access password.";
      authPassword.focus();
      return;
    }
    if (provided === ACCESS_PASSWORD) {
      sessionStorage.setItem(ACCESS_KEY, "1");
      unlockApp();
      authPassword.value = "";
    } else {
      authError.textContent = "Incorrect password.";
      authPassword.value = "";
      authPassword.focus();
    }
  });
}

const updateProgress = (fraction) => {
  if (!trainProgress || !trainProgressLabel) {
    return;
  }
  const percent = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  trainProgress.style.width = `${percent}%`;
  trainProgressLabel.textContent = `${percent}%`;
};

const refreshPlot = async () => {
  if (!lossPlot) return;
  try {
    const response = await fetch("/train/plot", { method: "HEAD" });
    if (response.ok) {
      lossPlot.src = `/train/plot?ts=${Date.now()}`;
      lossPlot.classList.remove("hidden");
    }
  } catch {
    // ignore
  }
};

const fetchTrainStatus = async () => {
  if (!trainStatus) {
    return;
  }
  try {
    const response = await fetch("/train/status");
    if (!response.ok) {
      throw new Error("Failed to fetch status");
    }
    const data = await response.json();
    trainStatus.textContent = data.message || "Unknown";
    updateProgress(data.progress || 0);
    if (data.running) {
      trainStatus.classList.add("running");
    } else {
      trainStatus.classList.remove("running");
      if (data.returncode === 0) {
        refreshPlot();
      }
    }
  } catch (err) {
    trainStatus.textContent = err.message;
  }
};

const fetchTrainLogs = async () => {
  if (!trainLogs) {
    return;
  }
  try {
    const response = await fetch("/train/logs");
    if (!response.ok) {
      throw new Error("Failed to fetch logs");
    }
    const data = await response.json();
    if (data.logs && data.logs.length > 0) {
      trainLogs.textContent = data.logs.join("\n");
    } else {
      trainLogs.textContent = "No logs yet.";
    }
  } catch (err) {
    trainLogs.textContent = err.message;
  }
};

const fetchDatasetStatus = async () => {
  if (!buildStatus) {
    return;
  }
  try {
    const response = await fetch("/dataset/status");
    if (!response.ok) {
      throw new Error("Failed to fetch dataset status");
    }
    const data = await response.json();
    buildStatus.textContent = data.message || "Idle";
    if (data.running) {
      buildStatus.classList.add("running");
    } else {
      buildStatus.classList.remove("running");
    }
  } catch (err) {
    buildStatus.textContent = err.message;
  }
};

const fetchModelStatus = async () => {
  if (!modelStatus) return;
  try {
    const response = await fetch("/model/status");
    if (!response.ok) {
      throw new Error("Failed to fetch model status");
    }
    const data = await response.json();
    if (data.exists) {
      const sizeMB = data.size ? (data.size / (1024 * 1024)).toFixed(2) : "0";
      modelStatus.textContent = `Model loaded (${data.filename || "model"} · ${sizeMB} MB)`;
    } else {
      modelStatus.textContent = "No model file available.";
    }
  } catch (err) {
    modelStatus.textContent = err.message;
  }
};

if (trainButton) {
  trainButton.addEventListener("click", async () => {
    trainButton.disabled = true;
    trainButton.textContent = "Starting...";
    try {
      let epochs = parseInt(trainEpochsInput?.value || "5", 10);
      if (Number.isNaN(epochs) || epochs < 1) {
        epochs = 5;
      }
      const payload = { epochs };
      const response = await fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      trainStatus.textContent = data.message || "requested";
    } catch (err) {
      trainStatus.textContent = err.message;
    } finally {
      trainButton.disabled = false;
      trainButton.textContent = "Start Training";
    }
    fetchTrainStatus();
  });
  fetchTrainStatus();
  fetchTrainLogs();
  refreshPlot();
  setInterval(() => {
    fetchTrainStatus();
    fetchTrainLogs();
  }, 5000);
}

if (buildButton) {
  buildButton.addEventListener("click", async () => {
    buildButton.disabled = true;
    buildButton.textContent = "Building...";
    if (buildStatus) buildStatus.textContent = "Starting dataset build...";
    try {
      const payload = {
        geometry: parseInt(geometryCountInput?.value || "200", 10),
        charts: parseInt(chartCountInput?.value || "200", 10),
        ocr: parseInt(ocrCountInput?.value || "50", 10),
        lines: parseInt(lineCountInput?.value || "40", 10),
        text: parseInt(textCountInput?.value || "400", 10),
      };
      const response = await fetch("/dataset/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (buildStatus) buildStatus.textContent = data.message || "dataset build requested";
    } catch (err) {
      if (buildStatus) buildStatus.textContent = err.message;
    } finally {
      buildButton.disabled = false;
      buildButton.textContent = "Build Dataset";
      fetchDatasetStatus();
    }
  });
  fetchDatasetStatus();
  setInterval(fetchDatasetStatus, 7000);
}

if (downloadModelBtn) {
  downloadModelBtn.addEventListener("click", () => {
    window.open("/model/download", "_blank");
  });
}

if (uploadModelBtn && uploadModelInput) {
  uploadModelBtn.addEventListener("click", async () => {
    if (uploadModelInput.files.length === 0) {
      modelStatus.textContent = "Select a model file first.";
      return;
    }
    const formData = new FormData();
    formData.append("file", uploadModelInput.files[0]);
    try {
      const response = await fetch("/model/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      modelStatus.textContent = data.message || "Upload complete.";
      uploadModelInput.value = "";
    } catch (err) {
      modelStatus.textContent = err.message;
    }
    fetchModelStatus();
  });
}

if (deleteModelBtn) {
  deleteModelBtn.addEventListener("click", async () => {
    try {
      const response = await fetch("/model", { method: "DELETE" });
      const data = await response.json();
      modelStatus.textContent = data.message || "Model deleted.";
    } catch (err) {
      modelStatus.textContent = err.message;
    }
    fetchModelStatus();
  });
}

fetchModelStatus();
setInterval(fetchModelStatus, 10000);


if (predictForm) {
  predictForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    output.textContent = "Running prediction...";
    const formData = new FormData();
    const prompt = document.getElementById("prompt").value.trim();
    const fileInput = document.getElementById("image");
    formData.append("prompt", prompt);
    if (fileInput.files.length > 0) {
      formData.append("file", fileInput.files[0]);
    }
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Prediction failed");
      }
      const data = await response.json();
      output.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      output.textContent = err.message;
    }
  });
}
