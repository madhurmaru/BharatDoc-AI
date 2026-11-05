# BharatDoc AI

A completely local document understanding mini-project. The Next.js interface and FastAPI backend run on your Mac; documents and model files stay on the attached SSD.

## What it does

- Upload PDF, PNG, and JPG files (25 MB, 20 pages maximum).
- Extract embedded PDF text; use local Tesseract OCR for scanned pages and images. Select the document language when uploading. English and eleven Indian-language OCR packs are included on the SSD.
- Extract dates, email addresses, phone numbers, PIN codes, IFSC codes, and rupee amounts.
- Summarize documents and answer questions with cited source pages using a local language model and SQLite full-text retrieval.
- Translate across major Indian languages using a locally stored translation model.
- Review document counts, processing time, and measured OCR and field accuracy.

## Start on this Mac

Keep the SSD connected, then double-click `start.command`. It starts the app and opens http://127.0.0.1:3000. Leave the Terminal window open while using the app; press Ctrl+C there to stop it.

## Rebuild if needed

Run these commands from the project folder on the SSD:

```sh
./backend/setup-vision.sh
cd frontend && npm install && cd ..
```

The launcher keeps model and temporary caches on the SSD and forces local-only model loading. You can also start the services separately in Terminal:

Start the backend in one terminal:

```sh
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another:

```sh
cd frontend && npm run dev
```

Open http://127.0.0.1:3000. After setup, no cloud API is used. Model loading uses local files only.

## Models

The app uses [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) from `models/qwen-1.5b` for summary and chat, and [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) from `models/nllb` for fallback translation. NLLB is licensed CC-BY-NC-4.0 and is intended for research use, fitting this college mini-project. Translation uses a short-lived CPU worker because it loaded faster and released memory after each request. The official [AI4Bharat IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) models are preferred but gated; access has not been granted on this Mac. [BharatGen Patram](https://huggingface.co/bharatgenai/patram-7b-instruct) is also gated and its published full-precision weights are too large to assume practical use with 16 GB shared RAM. Those integrations remain pending until access and compatibility can be verified.

The project, models, OCR data, uploads, database, and caches are on the external SSD. It uses Python, Node.js, Git, and Poppler already installed on the Mac's internal drive; these shared system tools were not duplicated on the SSD.

AI summaries can misread ambiguous dates or infer details. Always check important answers against the extracted page text.

## Evaluation

The dashboard records character error rate (CER), word error rate (WER), and exact field-set accuracy from reference text and fields entered by the user. A lower CER or WER is better; a higher field accuracy is better. These metrics are only meaningful when reference answers are supplied.
