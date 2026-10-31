/**
 * EmailUploadInterface
 * --------------------
 * Handles user interaction on the "Gone Phishing" page:
 *   - lets the user select an .eml file
 *   - submits it to the backend when "Analyze" is clicked
 *   - shows an error if no file / an unsupported file is attached
 *   - displays the analysis result returned by the backend
 *
 * Attributes (snake_case kept to match the class design):
 *   this.selected_file  -> File | null   the file the user picked
 *   this.error_message  -> string        the current error ("" if none)
 *   this.result         -> object | null the last analysis result
 *
 * Expected result shape from the backend (future AnalysisResult):
 *   {
 *     classification: "safe" | "phishing",
 *     safety_score: 0-100,
 *     explanation: string | null,
 *     explanation_available: boolean
 *   }
 */
class EmailUploadInterface {       /* defines the EmailUploadInterface class */ 
  static ALLOWED_EXTENSION = ".eml";
  static NO_FILE_MESSAGE = "Please attach a valid .eml file before clicking Analyze.";
  static WRONG_TYPE_MESSAGE = "Unsupported file type. Please attach a valid .eml file.";
  static EMPTY_FILE_MESSAGE = "The attached file is empty. Please attach a valid .eml file.";

  constructor({ fileInput, fileNameLabel, analyzeButton, errorBox, resultBox, endpoint = "/api/analyze" }) {
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
    this.endpoint = endpoint;

    // Wire up user events
    this.fileInput.addEventListener("change", (event) => {
      const file = event.target.files.length > 0 ? event.target.files[0] : null;
      this.select_file(file);
    });
    this.analyzeButton.addEventListener("click", () => this.submit_email());
  }

  /** Store the file the user picked and clear any old error/result. */
  select_file(file) {
    this.selected_file = file || null;
    this.error_message = "";
    this.result = null;

    this.fileNameLabel.textContent = this.selected_file ? this.selected_file.name : "No file selected";
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

    const scoreLine = document.createElement("p");
    scoreLine.className = "result__score";
    scoreLine.textContent = `Safety score: ${score} / 100`;

    const summary = document.createElement("p");
    summary.textContent = `The email is ${verdict} with a score of ${score}.`;

    this.resultBox.append(heading, scoreLine, summary);

    // Explanations are only requested for phishing emails
    if (!isSafe) {
      const detail = document.createElement("p");
      if (result.explanation_available && result.explanation) {
        detail.className = "result__explanation";
        detail.textContent = result.explanation;
      } else {
        detail.className = "result__notice";
        detail.textContent = "The AI explanation service is currently unavailable, so a detailed explanation can't be shown.";
      }
      this.resultBox.append(detail);
    }

    this.#show(this.resultBox);
  }

  // ---- private helpers ----
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
  });
});
