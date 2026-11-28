/**
 * EmailUploadInterface
 * --------------------
 * Handles user interaction on the "Gone Phishing" page:
 *   - lets the user select an .eml file
 *   - submits it to the backend when "Analyze" is clicked
 *   - shows an error if no file / an unsupported file is attached
 *   - displays the analysis result returned by the backend
 *   }
 */
class EmailUploadInterface {
  static ALLOWED_EXTENSION = ".eml";
  static NO_FILE_MESSAGE = "Please attach a valid .eml file before clicking Analyze.";
  static WRONG_TYPE_MESSAGE = "Unsupported file type. Please attach a valid .eml file.";
  static EMPTY_FILE_MESSAGE = "The attached file is empty. Please attach a valid .eml file.";

  constructor({ fileInput, fileNameLabel, analyzeButton, errorBox, resultBox, dropZone = null, endpoint = "/api/analyze" }) {
    // Attributes from the class design
    this.selected_file = null;
    this.error_message = "";
    this.result = null;

    // DOM elements this class controls
    this.fileInput = fileInput;
    this.fileNameLabel = fileNameLabel;
    this.analyzeButton = analyzeButton;
    this.errorBox = errorBox;
    this.resultBox = resultBox;
    this.dropZone = dropZone;
    this.endpoint = endpoint;

    // Wire up user events
    this.fileInput.addEventListener("change", (event) => {
      const file = event.target.files.length > 0 ? event.target.files[0] : null;
      this.select_file(file);
    });
    this.analyzeButton.addEventListener("click", () => this.submit_email());
    if (this.dropZone) this.#enableDragAndDrop();
  }

  /** Store the file the user picked and clear any old error/result. */
  select_file(file) {
    this.selected_file = file || null;
    this.error_message = "";
    this.result = null;

    this.fileNameLabel.textContent = this.selected_file ? this.selected_file.name : "No file selected";
    if (this.dropZone) this.dropZone.classList.toggle("has-file", Boolean(this.selected_file));
    this.#hide(this.errorBox);
    this.#hide(this.resultBox);
  }

  /** Validate the selection, then send it to the backend for analysis. */
  async submit_email() {
    // 1. Client-side checks (the backend EmailValidator will re-check later)
    if (!this.selected_file) {
      this.display_error(EmailUploadInterface.NO_FILE_MESSAGE);
      return;
    }
    if (!this.selected_file.name.toLowerCase().endsWith(EmailUploadInterface.ALLOWED_EXTENSION)) {
      this.display_error(EmailUploadInterface.WRONG_TYPE_MESSAGE);
      return;
    }
    if (this.selected_file.size === 0) {
      this.display_error(EmailUploadInterface.EMPTY_FILE_MESSAGE);
      return;
    }

    // 2. Send to backend
    const formData = new FormData();
    formData.append("file", this.selected_file);
    this.#setLoading(true);

    try {
      const response = await fetch(this.endpoint, { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        this.display_error(data.detail || "The server could not analyze this file. Please try again.");
        return;
      }
      this.display_result(data);
    } catch (err) {
      this.display_error("Could not reach the server. Please check that the backend is running.");
    } finally {
      this.#setLoading(false);
    }
  }

  /** Show an error message and hide any previous result. */
  display_error(message) {
    this.error_message = message;
    this.result = null;

    this.errorBox.textContent = message;
    this.#show(this.errorBox);
    this.#hide(this.resultBox);
  }

  /** Render the analysis result (with a fallback if the AI explanation is unavailable). */
  display_result(result) {
    this.result = result;
    this.error_message = "";
    this.#hide(this.errorBox);

    const isSafe = result.classification === "safe";
    const score = Math.round(Number(result.safety_score));
    const verdict = isSafe ? "safe" : "unsafe";

    // Build with DOM nodes + textContent so nothing from the server is ever run as HTML
    this.resultBox.replaceChildren();
    this.resultBox.className = `result ${isSafe ? "result--safe" : "result--unsafe"}`;

    const heading = document.createElement("h2");
    heading.textContent = isSafe ? "Likely safe" : "Likely phishing";

    // Big score number
    const scoreRow = document.createElement("div");
    scoreRow.className = "result__score-row";
    const scoreNumber = document.createElement("span");
    scoreNumber.className = "result__score-number";
    scoreNumber.textContent = String(score);
    const scoreLabel = document.createElement("span");
    scoreLabel.className = "result__score-label";
    scoreLabel.textContent = "/ 100 safety score";
    scoreRow.append(scoreNumber, scoreLabel);

    // Meter: red when dangerous, yellow in the middle, green when safe
    const meter = document.createElement("div");
    meter.className = "meter";
    meter.setAttribute("role", "img");
    meter.setAttribute("aria-label", `Safety score ${score} out of 100`);
    const fill = document.createElement("div");
    const level = score < 40 ? "low" : score < 70 ? "mid" : "high";
    fill.className = `meter__fill meter__fill--${level}`;
    fill.style.width = `${Math.min(Math.max(score, 0), 100)}%`;
    meter.append(fill);

    const scale = document.createElement("div");
    scale.className = "meter__scale";
    const scaleLow = document.createElement("span");
    scaleLow.textContent = "0 · phishing";
    const scaleHigh = document.createElement("span");
    scaleHigh.textContent = "100 · safe";
    scale.append(scaleLow, scaleHigh);

    const summary = document.createElement("p");
    summary.className = "result__summary";
    summary.textContent = `The email is ${verdict} with a score of ${score}.`;

    this.resultBox.append(heading, scoreRow, meter, scale, summary);

    // Explanations are only requested for phishing emails
    if (!isSafe) {
      const detail = document.createElement("p");
      if (result.explanation_available && result.explanation) {
        detail.className = "result__explanation";
        const title = document.createElement("span");
        title.className = "result__explanation-title";
        title.textContent = "Why this looks unsafe";
        detail.append(title, document.createElement("br"), result.explanation);
      } else {
        detail.className = "result__notice";
        detail.textContent = "The AI explanation service is currently unavailable, so a detailed explanation can't be shown.";
      }
      this.resultBox.append(detail);
    }

    this.#show(this.resultBox);
  }

  // ---- private helpers ----
  #enableDragAndDrop() {
    // Stop the browser from opening a file dropped anywhere else on the page
    for (const type of ["dragover", "drop"]) {
      window.addEventListener(type, (event) => event.preventDefault());
    }
    for (const type of ["dragenter", "dragover"]) {
      this.dropZone.addEventListener(type, (event) => {
        event.preventDefault();
        this.dropZone.classList.add("is-dragging");
      });
    }
    this.dropZone.addEventListener("dragleave", (event) => {
      if (!this.dropZone.contains(event.relatedTarget)) this.dropZone.classList.remove("is-dragging");
    });
    this.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      this.dropZone.classList.remove("is-dragging");
      const files = event.dataTransfer ? event.dataTransfer.files : null;
      this.select_file(files && files.length > 0 ? files[0] : null);
    });
  }

  #show(el) { el.hidden = false; }
  #hide(el) { el.hidden = true; }
  #setLoading(isLoading) {
    this.analyzeButton.disabled = isLoading;
    this.analyzeButton.textContent = isLoading ? "Analyzing…" : "Analyze";
  }
}

// Create the interface once the page has loaded
document.addEventListener("DOMContentLoaded", () => {
  window.emailUploadInterface = new EmailUploadInterface({
    fileInput: document.getElementById("email-file"),
    fileNameLabel: document.getElementById("file-name"),
    analyzeButton: document.getElementById("analyze-btn"),
    errorBox: document.getElementById("error-box"),
    resultBox: document.getElementById("result-box"),
    dropZone: document.getElementById("drop-zone"),
  });
});