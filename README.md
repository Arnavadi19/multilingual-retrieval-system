Here is a clean, professional **“Frontend Setup & Running Instructions”** section you can add directly to your GitHub README. It matches the structure and tone of the rest of your documentation.

---

## Frontend (React UI) Setup & Running

This project includes an optional React-based frontend for interacting with the Multilingual Information Retrieval API. The backend runs on **FastAPI + Uvicorn**, and the frontend runs on **React + Vite**.

### 🖥️ Backend API (FastAPI)

Before starting the frontend, make sure the backend API is running.

```bash
python -m uvicorn main_api:app --reload --port 8000
```

* Starts the FastAPI server on **[http://localhost:8000](http://localhost:8000)**
* The `/search` endpoint is used by the frontend to retrieve results
* The `--reload` flag auto-restarts the server on code changes

---

## 🌐 Frontend (React + Vite UI)

### 1. Navigate to the Frontend Directory

```bash
cd multilingual-retrieval-ui
```

You must be inside this directory to run any Node / npm commands.

### 2. Install Dependencies

```bash
npm install
```

Installs all required packages defined in `package.json` (React, Vite, Axios, UI libraries, etc.).

### 3. Start the Development Server

```bash
npm run dev
```

* Launches the Vite development server
* The default app URL is:

  ```
  http://localhost:5173
  ```
* The frontend automatically sends search queries to the backend at **[http://localhost:8000](http://localhost:8000)** (configurable in `.env` or Axios instance).

---

## 🧩 Frontend–Backend Integration

The React UI communicates with the FastAPI backend through REST API calls.

Typical configuration inside your UI (e.g., `src/config.js` or Axios instance):

```javascript
export const API_BASE_URL = "http://localhost:8000";
```

The frontend makes requests like:

```javascript
axios.get(`${API_BASE_URL}/search`, {
    params: { query, top_k }
})
```

Ensure both servers are running:

| Service             | Command                                               | Default URL                                    |
| ------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| **Backend API**     | `python -m uvicorn main_api:app --reload --port 8000` | [http://localhost:8000](http://localhost:8000) |
| **Frontend (Vite)** | `npm run dev`                                         | [http://localhost:5173](http://localhost:5173) |

---

## 📦 Optional: Build Frontend for Production

To prepare the frontend for deployment:

```bash
npm run build
```

This generates an optimized production build in the `dist/` folder.
