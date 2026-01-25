"""Debug routes (e.g. list registered routes)."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/debug/routes")
async def list_routes(request: Request):
    app = request.app
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, "name", "unknown"),
            })
    return {"routes": routes, "total": len(routes)}
