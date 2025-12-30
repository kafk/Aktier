#!/usr/bin/env python3
"""Run the Stock News Monitor application."""

import uvicorn

if __name__ == "__main__":
    print("Starting Stock News Monitor...")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop")
    print("-" * 40)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
