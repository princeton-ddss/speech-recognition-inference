import uvicorn

def launch():
    from __init__ import config

    if __name__ == '__main__':
        uvicorn.run(
            "main:app", port=config.port, host=config.host,
            reload=config.auto_reload
        )

