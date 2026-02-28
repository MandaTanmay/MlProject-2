"""
Rate Limiting Middleware
Implements simple in-memory rate limiting for production API.
Limits requests per IP address to prevent abuse.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from collections import defaultdict
import time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware using in-memory storage.
    For production, consider Redis-based rate limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute per IP
            requests_per_hour: Max requests per hour per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Storage: {ip: [(timestamp, count), ...]}
        self.request_history = defaultdict(list)
        
        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware
            
        Returns:
            Response or HTTPException
        """
        # Get client IP
        client_ip = request.client.host
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/full", "/"]:
            return await call_next(request)
        
        # Current time
        now = time.time()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
            self.last_cleanup = now
        
        # Get request history for this IP
        history = self.request_history[client_ip]
        
        # Remove old entries (older than 1 hour)
        cutoff_time = now - 3600
        history = [entry for entry in history if entry > cutoff_time]
        self.request_history[client_ip] = history
        
        # Check rate limits
        
        # 1. Check requests per minute
        minute_ago = now - 60
        requests_last_minute = sum(1 for timestamp in history if timestamp > minute_ago)
        
        if requests_last_minute >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute. Try again later.",
                headers={"Retry-After": "60"}
            )
        
        # 2. Check requests per hour
        if len(history) >= self.requests_per_hour:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.requests_per_hour} requests per hour. Try again later.",
                headers={"Retry-After": "3600"}
            )
        
        # Add current request to history
        history.append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, self.requests_per_minute - requests_last_minute - 1))
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, self.requests_per_hour - len(history)))
        
        return response
    
    def _cleanup_old_entries(self):
        """Remove old entries from request history to prevent memory bloat."""
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        # Clean up old IPs completely
        ips_to_remove = []
        for ip, history in self.request_history.items():
            # Remove old timestamps
            history = [ts for ts in history if ts > cutoff_time]
            
            if not history:
                ips_to_remove.append(ip)
            else:
                self.request_history[ip] = history
        
        # Remove empty IPs
        for ip in ips_to_remove:
            del self.request_history[ip]
        
        print(f"[Rate Limiter] Cleanup: Removed {len(ips_to_remove)} old IPs, tracking {len(self.request_history)} active IPs")
