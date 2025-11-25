# Backend Setup Guide

This guide will help you set up and run the backend server for the project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**MacOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

First, try installing from the requirements file:

```bash
pip install -r requirements.txt
```

If you encounter errors, install the core dependencies manually:

```bash
pip install uvicorn fastapi firebase-admin python-dotenv sqlmodel 'pydantic[email]' aiosqlite boto3 python-multipart
```

> **Note:** If any additional errors occur, install the specific missing library using:
> ```bash
> pip install <library-name>
> ```

### 4. Configure Environment Variables

1. Copy the `.env.example` file and rename it to `.env`
2. Fill in the required environment variables in the `.env` file

### 5. Setup Firebase

Create a `service-account.json` file in the project root with your Firebase service account credentials.

## Running the Server

Start the development server with hot reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://127.0.0.1:8000`

### API Documentation

Once the server is running, you can access the interactive API documentation at:

- **Swagger UI:** http://127.0.0.1:8000/docs

## Database

The application will automatically create a local SQLite database (`dev.db`) on first run.

## Troubleshooting

- **Virtual environment issues:** Delete the `venv` folder and recreate it
- **Module not found errors:** Install the missing module with `pip install <module-name>`
- **Firebase errors:** Verify your `service-account.json` is correctly placed and formatted

## Project Structure

```
inter-backend/
├── venv/                 # Virtual environment (excluded from git)
├── main.py              # Main application file
├── dev.db              
├── app/
├── utils/
├── config.py
├── render.yaml
├── service-account.json # Firebase credentials (excluded from git)
├── .env                 # Environment variables (excluded from git)
├── .env.example         # Example environment file
├── requirements.txt     
└── README.md           
```