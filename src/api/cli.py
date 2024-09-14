import uvicorn


def launch():
    from api import config

    uvicorn.run(
        "api:app", port=config.port, host=config.host, reload=config.auto_reload
    )
