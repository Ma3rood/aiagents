# ConvertClick Backend

This is the backend service for ConvertClick, built with FastAPI, MongoDB, and Redis.

## Prerequisites

- Python 3.8+
- MongoDB
- Redis
- Poetry (optional but recommended for dependency management)

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment file and configure it:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration values.

5. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`. This is an API-only backend service.

## API Usage

- API routes are prefixed with `/api/v1`
- The root endpoint `/` provides API information
- CORS is enabled to allow frontend applications to access the API

## API Documentation

Once the application is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Project Structure

```
Backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       └── router.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   └── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Health Checks

- `GET /api/v1/health`: Basic health check
- `GET /api/v1/health/db`: Database connectivity check 