"""
Test Target Endpoint - For WAF Testing

This endpoint is NOT in the /api/* path, so it will go through the WAF middleware.
Use this to generate real traffic that appears in the dashboard.
"""
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/test/endpoint")
async def test_endpoint(
    request: Request,
    id: str = Query(None),
    search: str = Query(None),
    user: str = Query(None)
):
    """Test endpoint that goes through WAF middleware"""
    return JSONResponse({
        "success": True,
        "message": "Test endpoint - if you see this, the request was allowed by WAF",
        "params": {
            "id": id,
            "search": search,
            "user": user
        }
    })


@router.post("/test/login")
async def test_login(request: Request):
    """Test login endpoint"""
    try:
        body = await request.json()
    except:
        body = {}

    return JSONResponse({
        "success": True,
        "message": "Test login endpoint",
        "received": body
    })


@router.get("/test/profile")
async def test_profile(name: str = Query(None)):
    """Test profile endpoint"""
    return JSONResponse({
        "success": True,
        "message": "Test profile endpoint",
        "name": name
    })


@router.post("/test/search")
async def test_search(request: Request):
    """Test search endpoint"""
    try:
        body = await request.json()
    except:
        body = {}

    return JSONResponse({
        "success": True,
        "message": "Test search endpoint",
        "query": body.get("query")
    })
