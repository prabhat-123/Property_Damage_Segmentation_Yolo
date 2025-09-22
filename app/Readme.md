✅ How this works

Build the base image first:
    ```
    docker build -t crack-seg-base -f Dockerfile.base .
    ```

Build and run services with Compose:
    ```
    docker compose up --build
    ```

Streamlit depends on FastAPI → Compose waits for FastAPI container to start before starting Streamlit.
Both services share the same dependencies installed in the base image, so you don’t reinstall twice.
