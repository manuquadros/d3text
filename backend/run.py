import uvicorn


def main() -> None:
    uvicorn.run("api:app", reload=True)
