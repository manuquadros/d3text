import uvicorn


def main() -> None:
    uvicorn.run("api:app", reload=True, reload_excludes="test_*.py")
