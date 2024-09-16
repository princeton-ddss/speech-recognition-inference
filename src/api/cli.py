import uvicorn

def launch():
    from .__init__ import config

    uvicorn.run(
        "api.main:app", port=config.port, host=config.host,
        reload=config.auto_reload
    )

