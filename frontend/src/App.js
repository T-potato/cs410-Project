import React, { useState } from "react";
import axios from "axios";
import * as pdfjs from "pdfjs-dist/build/pdf";

// Configure the PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

function App() {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState("bart");
  const [maxSummaryLength, setMaxSummaryLength] = useState(150);

  const handleTextInputChange = (e) => {
    setText(e.target.value);
    setPdfFile(null); // Reset PDF file if text input is used
  };

  const handlePdfUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setPdfFile(file);
      setText(""); // Reset text input if PDF is uploaded

      try {
        const fileReader = new FileReader();
        fileReader.onload = async () => {
          const typedArray = new Uint8Array(fileReader.result);
          const pdf = await pdfjs.getDocument(typedArray).promise;
          let extractedText = "";

          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map((item) => item.str).join(" ");
            extractedText += pageText + "\n";
          }

          setText(extractedText);
        };
        fileReader.readAsArrayBuffer(file);
      } catch (error) {
        console.error("Error parsing PDF: ", error);
        alert("Failed to parse the uploaded PDF.");
      }
    }
  };

  const handleSubmit = async () => {
    if (!text.trim()) {
      alert("Please provide input text or upload a PDF file.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000", {
        text: text,
        model_type: modelType,
        max_summary_length: maxSummaryLength,
      });
      setSummary(response.data.summary || "No summary received.");
    } catch (error) {
      console.error("Error sending request:", error);
      if (error.response?.data?.detail?.includes("Fine-tuned model not found")) {
        alert("Fine-tuned model not found. Please run train_summarizer.py first to use this option.");
      } else {
        alert("Failed to retrieve summary. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Text Summarization Application</h1>

      <div style={{ marginBottom: "20px" }}>
        <textarea
          rows="6"
          cols="60"
          placeholder="Enter text here..."
          value={text}
          onChange={handleTextInputChange}
          style={{ marginBottom: "10px", padding: "10px" }}
        />
        <br />
        <input type="file" accept=".pdf" onChange={handlePdfUpload} />
        <br />
        <div style={{ margin: "10px 0" }}>
          <label>
            Model Type:
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              style={{ marginLeft: "10px" }}
            >
              <option value="bart">BART</option>
              <option value="bart-finetuned">BART (Fine-tuned)</option>
              <option value="t5">T5</option>
            </select>
          </label>
          <br />
          <br />
          <label>
            Max Summary Length:
            <input
              type="number"
              value={maxSummaryLength}
              onChange={(e) => setMaxSummaryLength(Number(e.target.value))}
              style={{ marginLeft: "10px" }}
            />
          </label>
        </div>
        <button
          onClick={handleSubmit}
          style={{
            marginTop: "20px",
            padding: "10px 20px",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          {loading ? "Summarizing..." : "Get Summary"}
        </button>
      </div>

      <div>
        <h2>Summary:</h2>
        <p
          style={{
            whiteSpace: "pre-wrap",
            padding: "10px",
            backgroundColor: "#f9f9f9",
            border: "1px solid #ddd",
            borderRadius: "5px",
          }}
        >
          {summary}
        </p>
      </div>
    </div>
  );
}

export default App;
