"""
Ponto de entrada — inicia o servidor FastAPI.
Uso: python run_server.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
