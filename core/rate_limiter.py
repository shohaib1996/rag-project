import time
from collections import defaultdict

# requests per window
RATE_LIMITS = {
    "ask": 20,  # 20 questions
    "train": 10,  # 10 training calls
}

WINDOW_SECONDS = 60  # per minute


class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    def check(self, user_id: str, action: str):
        now = time.time()
        window_start = now - WINDOW_SECONDS

        key = f"{user_id}:{action}"
        timestamps = self.requests[key]

        # remove old requests
        self.requests[key] = [ts for ts in timestamps if ts > window_start]

        if len(self.requests[key]) >= RATE_LIMITS[action]:
            return False

        self.requests[key].append(now)
        return True


rate_limiter = RateLimiter()
